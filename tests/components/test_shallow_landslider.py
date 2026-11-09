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


def make_linear_runout_grid():
    """Return a hand-routed row for exact progressive-deposition tests."""
    mg = RasterModelGrid((3, 7), xy_spacing=10.0)
    elevation = mg.add_zeros("topographic__elevation", at="node")
    soil = mg.add_zeros("soil__depth", at="node")
    receivers = mg.add_field(
        "hill_flow__receiver_node",
        np.repeat(np.arange(mg.number_of_nodes)[:, None], 2, axis=1),
        at="node",
    )
    proportions = mg.add_field(
        "hill_flow__receiver_proportions",
        np.zeros((mg.number_of_nodes, 2)),
        at="node",
    )
    proportions[:, 0] = 1.0

    path = np.array([8, 9, 10, 11, 12])
    receivers[path[:-1], 0] = path[1:]
    soil[path[0]] = 1.0
    return mg, path, elevation


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


def test_variable_cohesion_initializes_depth_weighted_strength_fields():
    mg = make_grid(add_soil=True)
    fraction = mg.add_zeros("soil__failed_material_fraction", at="node")
    fraction[mg.core_nodes] = np.linspace(0.0, 1.0, mg.core_nodes.size)

    comp = ShallowLandslider(
        mg,
        cohesion_eff=100.0,
        failed_material_cohesion=20.0,
    )

    expected = 100.0 * (1.0 - fraction) + 20.0 * fraction
    np.testing.assert_allclose(mg.at_node["soil__effective_cohesion"], expected)
    assert comp.results["failed_material_fraction"] is fraction
    assert comp.results["effective_cohesion"] is mg.at_node[
        "soil__effective_cohesion"
    ]


@pytest.mark.parametrize("failed_cohesion", [-1.0, 101.0, np.nan])
def test_variable_cohesion_rejects_invalid_failed_strength(failed_cohesion):
    mg = make_grid(add_soil=True)

    with pytest.raises(ValueError, match="failed_material_cohesion"):
        ShallowLandslider(
            mg,
            cohesion_eff=100.0,
            failed_material_cohesion=failed_cohesion,
        )


def test_variable_cohesion_rejects_invalid_material_fraction():
    mg = make_grid(add_soil=True)
    fraction = mg.add_zeros("soil__failed_material_fraction", at="node")
    fraction[mg.core_nodes[0]] = 1.1

    with pytest.raises(ValueError, match="between zero and one"):
        ShallowLandslider(
            mg,
            cohesion_eff=100.0,
            failed_material_cohesion=20.0,
        )


def test_stability_rereads_failed_fraction_and_uses_reduced_cohesion():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1_000.0,
        failed_material_cohesion=100.0,
        angle_int_frict=30.0,
    )
    comp._compute_stability()
    fos_intact = comp.results["factor_of_safety"].copy()
    critical_wetness_intact = comp.results["critical_relative_wetness"].copy()
    critical_acceleration_intact = comp.results["a_transient"].copy()

    mg.at_node["soil__failed_material_fraction"][mg.core_nodes] = 1.0
    comp._compute_stability()

    assert np.all(
        mg.at_node["soil__effective_cohesion"][mg.core_nodes] == 100.0
    )
    assert np.all(
        comp.results["factor_of_safety"][mg.core_nodes]
        < fos_intact[mg.core_nodes]
    )
    assert np.all(
        comp.results["critical_relative_wetness"][mg.core_nodes]
        < critical_wetness_intact[mg.core_nodes]
    )
    assert np.all(
        comp.results["a_transient"][mg.core_nodes]
        < critical_acceleration_intact[mg.core_nodes]
    )


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
    labels[1, 1:4] = [1, 2, 3]
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
    assert [item["label"] for item in metadata["group_details"]] == [1, 2, 3]


def test_hydrologic_selection_uses_group_mean_critical_wetness():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(mg, selection_method="hydrologic")
    labels = np.zeros(mg.shape, dtype=int)
    labels[1, 1:3] = 1
    labels[2, 1:3] = 2
    critical = np.full(mg.number_of_nodes, np.nan)
    critical[np.flatnonzero(labels.ravel() == 1)] = [0.1, 0.5]
    critical[np.flatnonzero(labels.ravel() == 2)] = [0.6, 0.8]

    weights, metadata = comp._generate_hydrologic_selection_weight(
        labels, critical
    )

    assert np.all(weights[labels == 1] == 1.0)
    assert np.all(weights[labels == 2] == 0.0)
    assert metadata["group_details"] == [
        {
            "label": 1,
            "mean_critical_wetness": pytest.approx(0.3),
            "selection_weight": 1.0,
        },
        {
            "label": 2,
            "mean_critical_wetness": pytest.approx(0.7),
            "selection_weight": 0.0,
        },
    ]


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
    assert "same mean critical wetness" in metadata["reason"]


def test_newmark_logistic_probability_increases_with_candidate_displacement():
    mg = make_grid(add_soil=True)
    comp = ShallowLandslider(
        mg,
        selection_method="newmark_logistic",
        time_shaking=2.0,
        random_seed=5000,
    )
    labels = np.zeros(mg.shape, dtype=int)
    labels[1, 1:3] = 1
    labels[2, 1:3] = 2
    comp._pga_h[:] = 0.5
    comp._a_transient = np.zeros(mg.number_of_nodes, dtype=float)
    critical = comp._a_transient.reshape(mg.shape)
    critical[1, 1:3] = 0.4 * 0.5 * comp.g
    critical[2, 1:3] = 0.1 * 0.5 * comp.g

    probabilities, metadata = comp._generate_newmark_logistic_probability(labels)

    low = probabilities[labels == 1][0]
    high = probabilities[labels == 2][0]
    assert 0.0 < low < high <= comp.newmark_logistic_max_probability
    assert np.all(probabilities[labels == 0] == 0.0)
    assert metadata["displacement_model"] == (
        "jibson_2007_critical_acceleration_ratio"
    )
    assert metadata["displacement_aggregation"] == "candidate_mean"
    assert metadata["calibrated_probability"] is False


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("newmark_logistic_intercept", np.nan),
        ("newmark_logistic_slope", 0.0),
        ("newmark_logistic_reference_displacement", 0.0),
        ("newmark_logistic_max_probability", 1.1),
    ],
)
def test_newmark_logistic_rejects_invalid_parameters(keyword, value):
    mg = make_grid(add_soil=True)

    with pytest.raises(ValueError, match=keyword):
        ShallowLandslider(mg, **{keyword: value})


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
    assert np.all(comp.results["runout_distance"][expected_sources] == 20.0)
    assert np.all(
        np.isnan(
            comp.results["runout_distance"][
                ~np.isin(np.arange(mg.number_of_nodes), expected_sources)
            ]
        )
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


def test_claessens_runout_calculates_one_distance_per_selected_group():
    mg = make_runout_grid()
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1.0,
        angle_int_frict=10.0,
        selection_method="all",
        random_seed=17,
        enable_runout=True,
        update_soil=True,
        runout_source_mode="selected",
        runout_distance_model="claessens",
        runout_claessens_parameters={
            "fraction": 0.4,
            "minimum_movement_slope": 10.0,
        },
        split_by_width_config=None,
    )

    comp.run_one_step()

    results = comp.results
    sources = results["runout"]["source_nodes"]
    labels = results["selected_labels"]
    distances = results["runout_distance"]
    head_elevations = results["runout_head_elevation"]
    start_found = results["runout_deposition_start_found"]
    start_nodes = results["runout_deposition_start_node"]
    start_elevations = results["runout_deposition_start_elevation"]
    erosional_distances = results["runout_erosional_path_distance"]
    vertical_drops = results["runout_vertical_drop"]
    deposition_reaches = results["runout_deposition_reach"]
    elevation = (
        mg.at_node["bedrock__elevation"] + results["input_soil_depth"]
    )

    assert sources.size > 0
    for label in np.unique(labels[sources]):
        group_sources = sources[labels[sources] == label]
        head_elevation = head_elevations[group_sources[0]]
        found = bool(start_found[group_sources[0]])
        if found:
            start_node = int(start_nodes[group_sources[0]])
            start_elevation = start_elevations[group_sources[0]]
            expected_drop = max(head_elevation - start_elevation, 0.0)
            expected_reach = 0.4 * expected_drop
            expected_distance = (
                erosional_distances[group_sources[0]] + expected_reach
            )
        else:
            start_node = np.nan
            start_elevation = np.nan
            expected_drop = np.nan
            expected_reach = np.nan
            expected_distance = erosional_distances[group_sources[0]]

        np.testing.assert_allclose(
            head_elevations[group_sources], np.max(elevation[group_sources])
        )
        np.testing.assert_allclose(start_found[group_sources], float(found))
        np.testing.assert_allclose(start_nodes[group_sources], start_node)
        np.testing.assert_allclose(start_elevations[group_sources], start_elevation)
        np.testing.assert_allclose(vertical_drops[group_sources], expected_drop)
        np.testing.assert_allclose(
            deposition_reaches[group_sources], expected_reach
        )
        np.testing.assert_allclose(distances[group_sources], expected_distance)

        group = results["group_properties"].loc[label]
        assert group["runout_head_elevation_m"] == pytest.approx(head_elevation)
        assert bool(group["runout_deposition_start_found"]) is found
        if found:
            assert group["runout_deposition_start_node"] == start_node
            assert group["runout_deposition_start_elevation_m"] == pytest.approx(
                start_elevation
            )
            assert group["runout_vertical_drop_m"] == pytest.approx(expected_drop)
            assert group["runout_deposition_reach_m"] == pytest.approx(
                expected_reach
            )
        else:
            assert np.isnan(group["runout_deposition_start_node"])
            assert np.isnan(group["runout_deposition_start_elevation_m"])
            assert np.isnan(group["runout_vertical_drop_m"])
            assert np.isnan(group["runout_deposition_reach_m"])
        assert group["runout_distance_m"] == pytest.approx(expected_distance)


def test_claessens_distance_uses_vertical_drop_at_slope_threshold():
    mg = make_runout_grid()
    comp = ShallowLandslider(
        mg,
        runout_distance_model="claessens",
        runout_claessens_parameters={
            "fraction": 0.4,
            "minimum_movement_slope": 10.0,
        },
    )
    head_node = int(mg.nodes.reshape(mg.shape)[2, 3])
    path = [head_node]
    for _ in range(2):
        receiver = comp._steepest_descent_receiver(path[-1])
        assert receiver is not None
        path.append(receiver)

    comp._slope_deg32[:] = 45.0
    comp._slope_deg32[path[-1]] = 5.0
    labels = np.zeros(mg.number_of_nodes, dtype=int)
    labels[head_node] = 1
    comp._selected_labels = labels

    distances = comp._resolve_claessens_runout_distance(
        np.array([head_node], dtype=int)
    )
    erosional_distance = sum(
        np.hypot(
            mg.node_x[second] - mg.node_x[first],
            mg.node_y[second] - mg.node_y[first],
        )
        for first, second in zip(path[:-1], path[1:])
    )
    vertical_drop = (
        mg.at_node["topographic__elevation"][head_node]
        - mg.at_node["topographic__elevation"][path[-1]]
    )

    assert comp._runout_deposition_start_found[head_node] == 1.0
    assert comp._runout_deposition_start_node[head_node] == path[-1]
    assert comp._runout_vertical_drop[head_node] == pytest.approx(vertical_drop)
    assert comp._runout_deposition_reach[head_node] == pytest.approx(
        0.4 * vertical_drop
    )
    assert distances[head_node] == pytest.approx(
        erosional_distance + 0.4 * vertical_drop
    )


def test_claessens_default_parameters_are_published_values():
    assert ShallowLandslider._validate_claessens_parameters(None) == {
        "fraction": 0.4,
        "minimum_movement_slope": 10.0,
    }


def test_claessens_auto_threshold_uses_bounded_landscape_percentile():
    mg = make_runout_grid()
    comp = ShallowLandslider(
        mg,
        runout_distance_model="claessens",
        runout_claessens_parameters={
            "minimum_movement_slope": "auto",
            "landscape_slope_percentile": 50.0,
            "landscape_slope_bounds": [8.0, 18.0],
        },
    )
    comp._slope_deg32[mg.core_nodes] = np.linspace(2.0, 42.0, mg.core_nodes.size)

    expected = np.clip(
        np.percentile(comp._slope_deg32[mg.core_nodes], 50.0), 8.0, 18.0
    )

    assert comp._resolve_claessens_slope_threshold() == pytest.approx(expected)
    assert comp.results["runout_claessens_slope_threshold"] == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"minimum_movement_slope": "regional"}, "number or 'auto'"),
        (
            {
                "minimum_movement_slope": "auto",
                "landscape_slope_percentile": 101.0,
            },
            "percentile",
        ),
        (
            {
                "minimum_movement_slope": "auto",
                "landscape_slope_bounds": [30.0, 5.0],
            },
            "bounds",
        ),
    ],
)
def test_claessens_auto_threshold_rejects_invalid_configuration(
    parameters, message
):
    with pytest.raises(ValueError, match=message):
        ShallowLandslider._validate_claessens_parameters(parameters)


def test_selected_runout_requires_distance_provider():
    mg = make_runout_grid()

    with pytest.raises(ValueError, match="requires runout_distance"):
        ShallowLandslider(
            mg,
            enable_runout=True,
            update_soil=True,
            runout_source_mode="selected",
        )


def test_claessens_runout_requires_valid_parameters():
    mg = make_runout_grid()
    with pytest.raises(ValueError, match="fraction cannot be negative"):
        ShallowLandslider(
            mg, runout_claessens_parameters={"fraction": -0.1}
        )

    with pytest.raises(ValueError, match=r"must be in \[0, 90\)"):
        ShallowLandslider(
            mg,
            runout_claessens_parameters={"minimum_movement_slope": 90.0},
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


def test_runout_marks_deposits_as_failed_material_and_reduces_cohesion():
    mg = make_runout_grid()
    soil_before = mg.at_node["soil__depth"].copy()
    intact_cohesion = 100.0
    failed_cohesion = 20.0
    comp = ShallowLandslider(
        mg,
        cohesion_eff=intact_cohesion,
        failed_material_cohesion=failed_cohesion,
        angle_int_frict=10.0,
        selection_method="all",
        enable_runout=True,
        update_soil=True,
        runout_source_mode="selected",
        runout_distance=20.0,
        split_by_width_config=None,
    )

    comp.run_one_step()

    deposition = mg.at_node["landslide__deposition"]
    erosion = mg.at_node["landslide__erosion"]
    final_depth = mg.at_node["soil__depth"]
    fraction = mg.at_node["soil__failed_material_fraction"]
    expected_fraction = np.zeros_like(final_depth)
    np.divide(
        deposition,
        final_depth,
        out=expected_fraction,
        where=final_depth > 0.0,
    )

    assert np.any(deposition > 0.0)
    np.testing.assert_allclose(fraction, expected_fraction)
    np.testing.assert_allclose(
        fraction * final_depth,
        deposition,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        final_depth,
        soil_before - erosion + deposition,
    )
    np.testing.assert_allclose(
        mg.at_node["soil__effective_cohesion"],
        intact_cohesion * (1.0 - fraction) + failed_cohesion * fraction,
    )
    assert np.all(
        mg.at_node["soil__effective_cohesion"][deposition > 0.0]
        < intact_cohesion
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


def test_bedrock_deposition_is_nonlinear_along_the_distance_limited_path():
    mg, path, elevation = make_linear_runout_grid()
    # Incoming material first reaches node 9. Its local gradient is 0.5, then
    # node 10 has gradient 0.25. With Sc=1, deposited fractions are 0.75 and
    # 0.9375 of the remaining mass, respectively. The flat node 11 traps the
    # residual.
    elevation[path] = [40.0, 30.0, 25.0, 22.5, 22.5]
    runout = ShallowLandslideRunout(mg)

    runout.run_one_step(
        source_nodes=np.array([path[0]]),
        runout_distance=100.0,
        deposition_model="bedrock",
        deposition_critical_slope=1.0,
    )

    deposition = runout.results["deposition"]
    assert deposition[path[1]] == pytest.approx(0.75)
    assert deposition[path[2]] == pytest.approx(0.25 * 0.9375)
    assert deposition[path[3]] == pytest.approx(0.25 * 0.0625)
    assert np.sum(runout.results["erosion"]) == pytest.approx(1.0)
    assert np.sum(deposition) == pytest.approx(1.0)
    assert runout.results["deposition_model"] == "bedrock"
    assert runout.results["deposition_critical_slope"] == 1.0
    assert runout.results["mass_balance_error"] == pytest.approx(0.0)
    assert all(
        error == pytest.approx(0.0)
        for error in runout.results["source_mass_balance_errors"].values()
    )


def test_bedrock_deposition_retains_residual_at_hard_distance_limit():
    mg, path, elevation = make_linear_runout_grid()
    elevation[path] = [50.0, 40.0, 30.0, 20.0, 10.0]
    runout = ShallowLandslideRunout(mg)

    runout.run_one_step(
        source_nodes=np.array([path[0]]),
        runout_distance=20.0,
        deposition_model="bedrock",
        deposition_critical_slope=0.5,
    )

    deposition = runout.results["deposition"]
    assert deposition[path[1]] == 0.0
    assert deposition[path[2]] == pytest.approx(1.0)
    assert np.sum(deposition) == pytest.approx(1.0)


@pytest.mark.parametrize("critical_slope", [None, 0.0, np.nan])
def test_bedrock_deposition_requires_positive_critical_slope(critical_slope):
    mg, path, _ = make_linear_runout_grid()
    runout = ShallowLandslideRunout(mg)

    with pytest.raises(ValueError, match="deposition_critical_slope"):
        runout.run_one_step(
            source_nodes=np.array([path[0]]),
            runout_distance=20.0,
            deposition_model="bedrock",
            deposition_critical_slope=critical_slope,
        )


def test_component_converts_bedrock_critical_slope_from_degrees():
    mg = make_runout_grid()
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1.0,
        angle_int_frict=30.0,
        selection_method="all",
        enable_runout=True,
        update_soil=True,
        runout_source_mode="selected",
        runout_distance=30.0,
        runout_deposition_model="bedrock",
        runout_deposition_parameters={"critical_slope": 45.0},
        split_by_width_config=None,
    )

    comp.run_one_step()

    assert comp.results["runout"]["deposition_model"] == "bedrock"
    assert comp.results["runout"]["deposition_critical_slope"] == pytest.approx(
        1.0
    )
    np.testing.assert_allclose(
        np.sum(comp.results["runout"]["erosion"]),
        np.sum(comp.results["runout"]["deposition"]),
    )


def test_progressive_deposits_conserve_failed_material_and_update_cohesion():
    mg, path, elevation = make_linear_runout_grid()
    elevation[path] = [40.0, 30.0, 25.0, 22.5, 22.5]
    mg.at_node["soil__depth"][path[1:]] = 1.0
    failed_fraction = mg.add_zeros(
        "soil__failed_material_fraction", at="node"
    )
    failed_fraction[path[1]] = 0.2
    comp = ShallowLandslider(
        mg,
        cohesion_eff=100.0,
        failed_material_cohesion=20.0,
    )
    runout = ShallowLandslideRunout(mg)
    soil_before = mg.at_node["soil__depth"].copy()
    fraction_before = failed_fraction.copy()

    runout.run_one_step(
        source_nodes=np.array([path[0]]),
        runout_distance=100.0,
        deposition_model="bedrock",
        deposition_critical_slope=1.0,
    )
    comp.update_failed_material_after_runout(
        soil_before,
        runout.results["erosion"],
        runout.results["deposition"],
    )

    erosion = runout.results["erosion"]
    deposition = runout.results["deposition"]
    final_soil = mg.at_node["soil__depth"]
    expected_failed_depth = (
        fraction_before * (soil_before - erosion) + deposition
    )
    np.testing.assert_allclose(failed_fraction * final_soil, expected_failed_depth)
    np.testing.assert_allclose(
        mg.at_node["soil__effective_cohesion"],
        100.0 * (1.0 - failed_fraction) + 20.0 * failed_fraction,
    )
    assert comp.results["failed_material_mass_balance_error"] == pytest.approx(
        0.0
    )
    assert np.sum(erosion) == pytest.approx(np.sum(deposition))


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


def test_aggregated_runout_matches_exhaustive_endpoint_transport():
    aggregated_grid = make_runout_grid()
    exhaustive_grid = make_runout_grid()
    aggregated = ShallowLandslideRunout(aggregated_grid)
    exhaustive = ShallowLandslideRunout(exhaustive_grid)
    sources = np.array([27, 28])
    distances = np.array([120.0, 150.0])

    aggregated.run_one_step(
        source_nodes=sources,
        runout_distance=distances,
    )
    exhaustive.run_one_step(
        source_nodes=sources,
        runout_distance=distances,
        return_paths=True,
    )

    np.testing.assert_allclose(
        aggregated.results["erosion"], exhaustive.results["erosion"]
    )
    np.testing.assert_allclose(
        aggregated.results["deposition"], exhaustive.results["deposition"]
    )
    np.testing.assert_allclose(
        aggregated_grid.at_node["soil__depth"],
        exhaustive_grid.at_node["soil__depth"],
    )
    assert sum(aggregated.results["source_path_counts"].values()) == len(
        exhaustive.results["paths"]
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
