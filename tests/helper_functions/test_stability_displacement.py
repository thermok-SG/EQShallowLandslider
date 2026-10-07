# tests/helper_functions/test_stability_displacement.py

import numpy as np
import pytest
from landlab import RasterModelGrid
from components.shallow_landslider import ShallowLandslider


def make_grid(n=5, spacing=10.0):
    mg = RasterModelGrid((n, n), xy_spacing=spacing)
    mg.add_ones("topographic__elevation", at="node")
    mg.add_ones("soil__depth", at="node")
    return mg


def test_factor_of_safety_matches_formula(monkeypatch):
    """Validate the FoS implementation against the analytic formula."""
    mg = make_grid()
    slope_rad = np.deg2rad(30.0)

    def constant_slope(elevs=None, **kwargs):
        slope = np.ones(mg.number_of_nodes) * slope_rad
        if kwargs.get("return_components", False):
            return slope, (slope, slope)
        return slope

    monkeypatch.setattr(mg, "calc_slope_at_node", constant_slope)
    monkeypatch.setattr(
        mg,
        "calc_aspect_at_node",
        lambda **kwargs: np.zeros(mg.number_of_nodes),
    )

    coh = 1000.0
    phi_rad = np.deg2rad(30.0)
    gamma_s = 15e3
    gamma_w = 9.8e3
    sub = 0.0

    comp = ShallowLandslider(
        mg,
        cohesion_eff=coh,
        angle_int_frict=30.0,
        submerged_soil_proportion=sub,
        update_soil=False,
    )

    fos = comp._factor_of_safety(
        mg, coh, phi_rad, submerged_soil_proportion=sub,
        soil_unit_weight=gamma_s, water_unit_weight=gamma_w
    )

    soil_depth = mg.at_node["soil__depth"]
    psi = sub * gamma_w * soil_depth
    slope = np.ones(mg.number_of_nodes) * slope_rad

    expected = ((coh - psi * np.tan(phi_rad)) /
                (gamma_s * soil_depth * np.sin(slope))) + \
               (np.tan(phi_rad) / np.tan(slope))

    assert np.allclose(fos, expected, rtol=1e-6, atol=1e-9)


def test_compute_stability_uses_configured_submerged_proportion(monkeypatch):
    """The public pipeline must forward m to the factor-of-safety equation."""
    mg = make_grid()
    slope = np.full(mg.number_of_nodes, np.deg2rad(30.0))
    monkeypatch.setattr(mg, "calc_slope_at_node", lambda **kwargs: slope)
    monkeypatch.setattr(
        mg, "calc_aspect_at_node", lambda **kwargs: np.zeros(mg.number_of_nodes)
    )
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1000.0,
        angle_int_frict=30.0,
        submerged_soil_proportion=0.8,
    )

    comp._compute_stability()
    expected = comp._factor_of_safety(
        mg,
        1000.0,
        np.deg2rad(30.0),
        submerged_soil_proportion=0.8,
    )

    assert np.allclose(comp.results["factor_of_safety"], expected, equal_nan=True)


def test_field_wetness_is_required_when_configured(monkeypatch):
    """Field mode must fail early instead of falling back to a constant."""
    mg = make_grid()
    monkeypatch.setattr(
        mg,
        "calc_aspect_at_node",
        lambda **kwargs: np.zeros(mg.number_of_nodes),
    )

    with pytest.raises(
        ValueError,
        match="soil__relative_wetness is required when wetness_source='field'",
    ):
        ShallowLandslider(mg, wetness_source="field")


@pytest.mark.parametrize("bad_value", [-0.01, 1.01, np.nan, np.inf])
def test_field_wetness_rejects_invalid_core_values(monkeypatch, bad_value):
    """Coupled hydrology errors must not be hidden by implicit clipping."""
    mg = make_grid()
    wetness = mg.add_zeros("soil__relative_wetness", at="node")
    wetness[:] = 0.5
    wetness[mg.core_nodes[0]] = bad_value
    monkeypatch.setattr(
        mg,
        "calc_aspect_at_node",
        lambda **kwargs: np.zeros(mg.number_of_nodes),
    )

    with pytest.raises(ValueError, match="soil__relative_wetness must be"):
        ShallowLandslider(mg, wetness_source="field")


def test_field_wetness_is_read_live_between_stability_evaluations(monkeypatch):
    """One component instance must respond to hydrology updates through time."""
    mg = make_grid()
    slope = np.full(mg.number_of_nodes, np.deg2rad(25.0))
    monkeypatch.setattr(mg, "calc_slope_at_node", lambda **kwargs: slope)
    monkeypatch.setattr(
        mg,
        "calc_aspect_at_node",
        lambda **kwargs: np.zeros(mg.number_of_nodes),
    )
    wetness = mg.add_zeros("soil__relative_wetness", at="node")
    wetness[:] = 0.30

    comp = ShallowLandslider(
        mg,
        cohesion_eff=1000.0,
        angle_int_frict=30.0,
        wetness_source="field",
    )
    comp._compute_stability()
    dry_fos = comp.results["factor_of_safety"].copy()
    assert np.all(dry_fos[mg.core_nodes] > 1.0)
    assert not np.any(comp.results["unstable_mask"][mg.core_nodes])

    wetness[:] = 0.60
    comp._compute_stability()
    wet_fos = comp.results["factor_of_safety"]

    assert np.all(wet_fos[mg.core_nodes] < dry_fos[mg.core_nodes])
    assert np.all(wet_fos[mg.core_nodes] < 1.0)
    assert np.all(comp.results["unstable_mask"][mg.core_nodes])
    assert np.shares_memory(comp.results["relative_wetness"], wetness)


def test_spatial_wetness_produces_spatial_stability(monkeypatch):
    """Node-wise wetness must not be collapsed to a domain statistic."""
    mg = make_grid()
    slope = np.full(mg.number_of_nodes, np.deg2rad(25.0))
    monkeypatch.setattr(mg, "calc_slope_at_node", lambda **kwargs: slope)
    monkeypatch.setattr(
        mg,
        "calc_aspect_at_node",
        lambda **kwargs: np.zeros(mg.number_of_nodes),
    )
    wetness = mg.add_zeros("soil__relative_wetness", at="node")
    wetness[mg.core_nodes] = np.linspace(0.1, 0.9, len(mg.core_nodes))

    comp = ShallowLandslider(
        mg,
        cohesion_eff=1000.0,
        angle_int_frict=30.0,
        wetness_source="field",
    )
    comp._compute_stability()

    core_fos = comp.results["factor_of_safety"][mg.core_nodes]
    assert np.all(np.diff(core_fos) < 0.0)
    np.testing.assert_allclose(
        comp.results["relative_wetness"][mg.core_nodes],
        wetness[mg.core_nodes],
    )


def test_constant_and_uniform_field_wetness_are_equivalent(monkeypatch):
    """The new field interface must preserve the established scalar equation."""
    slope_rad = np.deg2rad(25.0)

    def prepare_grid():
        grid = make_grid()
        monkeypatch.setattr(
            grid,
            "calc_slope_at_node",
            lambda **kwargs: np.full(grid.number_of_nodes, slope_rad),
        )
        monkeypatch.setattr(
            grid,
            "calc_aspect_at_node",
            lambda **kwargs: np.zeros(grid.number_of_nodes),
        )
        return grid

    constant_grid = prepare_grid()
    constant = ShallowLandslider(
        constant_grid,
        cohesion_eff=1000.0,
        angle_int_frict=30.0,
        submerged_soil_proportion=0.4,
        wetness_source="constant",
    )
    constant._compute_stability()

    field_grid = prepare_grid()
    field_grid.add_full("soil__relative_wetness", 0.4, at="node")
    spatial = ShallowLandslider(
        field_grid,
        cohesion_eff=1000.0,
        angle_int_frict=30.0,
        wetness_source="field",
    )
    spatial._compute_stability()

    for name in ("factor_of_safety", "a_transient", "a_driving", "a_diff"):
        np.testing.assert_allclose(
            constant.results[name], spatial.results[name], equal_nan=True
        )
    np.testing.assert_array_equal(
        constant.results["unstable_mask"], spatial.results["unstable_mask"]
    )


def test_critical_relative_wetness_matches_factor_of_safety_threshold(monkeypatch):
    """Substituting m_c into the unchanged equation must produce FoS == 1."""
    mg = make_grid()
    slope = np.full(mg.number_of_nodes, np.deg2rad(25.0))
    monkeypatch.setattr(mg, "calc_slope_at_node", lambda **kwargs: slope)
    monkeypatch.setattr(
        mg,
        "calc_aspect_at_node",
        lambda **kwargs: np.zeros(mg.number_of_nodes),
    )
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1000.0,
        angle_int_frict=30.0,
        pga_h=0.0,
        pga_v=0.0,
    )

    critical_wetness = comp._critical_relative_wetness(
        mg, 1000.0, np.deg2rad(30.0)
    )
    fos_at_threshold = comp._factor_of_safety(
        mg,
        1000.0,
        np.deg2rad(30.0),
        submerged_soil_proportion=critical_wetness,
    )

    np.testing.assert_allclose(fos_at_threshold[mg.core_nodes], 1.0, atol=1e-12)
    assert np.all((critical_wetness[mg.core_nodes] > 0.0))
    assert np.all((critical_wetness[mg.core_nodes] < 1.0))
    assert np.all(np.isnan(critical_wetness[mg.boundary_nodes]))


def test_zero_pga_critical_acceleration_matches_static_fos(monkeypatch):
    """Pin the identity that enables hydrologic failure without shaking."""
    mg = make_grid()
    slope = np.full(mg.number_of_nodes, np.deg2rad(25.0))
    monkeypatch.setattr(mg, "calc_slope_at_node", lambda **kwargs: slope)
    monkeypatch.setattr(
        mg,
        "calc_aspect_at_node",
        lambda **kwargs: np.zeros(mg.number_of_nodes),
    )
    wetness = mg.add_full("soil__relative_wetness", 0.6, at="node")
    comp = ShallowLandslider(
        mg,
        cohesion_eff=1000.0,
        angle_int_frict=30.0,
        wetness_source="field",
    )
    comp._compute_stability()

    expected = 9.81 * np.sin(slope) * (comp.results["factor_of_safety"] - 1.0)
    np.testing.assert_allclose(
        comp.results["a_transient"][mg.core_nodes],
        expected[mg.core_nodes],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(comp.results["a_driving"][mg.core_nodes] == 0.0)
    assert np.shares_memory(comp.results["relative_wetness"], wetness)


def test_critical_transient_acceleration(monkeypatch):
    mg = make_grid()
    slope_rad = np.deg2rad(20)
    def constant_slope(elevs=None, **kwargs):
        slope = np.ones(mg.number_of_nodes) * slope_rad
        if kwargs.get("return_components", False):
            return slope, (slope, slope)
        return slope

    monkeypatch.setattr(mg, "calc_slope_at_node", constant_slope)
    monkeypatch.setattr(
        mg,
        "calc_aspect_at_node",
        lambda **kwargs: np.zeros(mg.number_of_nodes),
    )

    g = 9.81
    a_h = np.ones(mg.number_of_nodes) * 0.3 * g
    a_v = np.ones(mg.number_of_nodes) * 0.1 * g

    phi = np.deg2rad(30)
    coh = 500.0
    gamma_s = 15e3
    gamma_w = 9.8e3
    sub = 0.0

    comp = ShallowLandslider(
        mg,
        cohesion_eff=coh,
        angle_int_frict=30,
        submerged_soil_proportion=sub,
    )
    ac, aslip, adiff = comp._critical_transient_acceleration(
        mg, coh, phi, sub, a_h=a_h, a_v=a_v,
        soil_unit_weight=gamma_s, water_unit_weight=gamma_w, g=g
    )

    soil_depth = mg.at_node["soil__depth"]
    psi = sub * gamma_w * soil_depth

    a_c_simple = (
        np.tan(phi) * (g * np.cos(slope_rad) - a_v * np.cos(slope_rad) - a_h * np.sin(slope_rad)) +
        ((g * coh) - (psi * g * np.tan(phi))) / (gamma_s * soil_depth) -
        g * np.sin(slope_rad)
    )
    a_c_simple[mg.boundary_nodes] = 0
    a_s = a_h * np.cos(slope_rad) - a_v * np.sin(slope_rad)

    assert np.allclose(ac, a_c_simple, rtol=1e-6)
    assert np.allclose(aslip, a_s)
    assert np.allclose(adiff, a_s - a_c_simple)


def test_newmark_displacement_and_masking():
    mg = make_grid()
    comp = ShallowLandslider(
        mg,
        cohesion_eff=10,
        angle_int_frict=30,
        compute_displacement=True,
    )

    # construct simple labels
    labels = np.zeros(mg.shape, dtype=int)
    labels[2:4, 2:4] = 1
    diff = np.zeros(mg.number_of_nodes)
    active_idx = np.where(labels.ravel() == 1)[0]
    diff[active_idx] = 2.0  # m/s2

    disp = comp._calculate_newmark_displacement(
        a_difference_1d=diff,
        selected_labels_2d=labels,
        time_shaking_2d=np.ones(mg.shape) * 3.0,
    )

    unlabeled = np.where(labels.ravel() == 0)[0]
    assert np.all(np.isnan(disp[unlabeled]))
    assert np.allclose(disp[active_idx], 0.5 * 2.0 * 9.0)

def test_newmark_displacement_threshold_behavior():
    mg = make_grid()
    comp = ShallowLandslider(
        mg, cohesion_eff=10, angle_int_frict=30,
        compute_displacement=True, displacement_threshold=5.0
    )

    comp.run_one_step()
    disp = comp.results["newmark"]

    # All high displacement nodes must exceed threshold
    for idx in comp._high_disp_nodes:
        assert disp[idx] > 5.0
        
def test_critical_acceleration_sets_boundary_to_zero(monkeypatch):
    mg = make_grid()
    comp = ShallowLandslider(mg, cohesion_eff=10, angle_int_frict=30)

    monkeypatch.setattr(
        mg, "calc_slope_at_node",
        lambda elevs=None, **kwargs: np.ones(mg.number_of_nodes) * np.deg2rad(10)
    )

    ac, *_ = comp._critical_transient_acceleration(
        mg, 10, np.deg2rad(30), 0.0,
        a_h=np.zeros(mg.number_of_nodes),
        a_v=np.zeros(mg.number_of_nodes)
    )

    assert np.all(ac[mg.boundary_nodes] == 0)
