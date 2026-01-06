
import numpy as np
import pytest
from landlab import RasterModelGrid

from helper_functions import (
    factor_of_safety,
    critical_transient_acceleration,
    calculate_newmark_displacement,
)


def make_grid(n=5, spacing=10.0):
    mg = RasterModelGrid((n, n), xy_spacing=spacing)
    # Soil depth: required by stability functions
    h = mg.add_zeros("soil__depth", at="node")
    h[:] = 1.0
    # Elevation: populated but overridden by monkeypatched slope
    z = mg.add_zeros("topographic__elevation", at="node")
    z[:] = 0.0
    return mg


def test_factor_of_safety_matches_formula(monkeypatch):
    mg = make_grid()
    # constant slope everywhere (30 deg)
    slope_rad = np.deg2rad(30.0)
    monkeypatch.setattr(
        mg, "calc_slope_at_node", lambda elevs=None: np.ones(mg.number_of_nodes) * slope_rad
    )

    cohesion_eff = 1000.0  # Pa
    phi = np.deg2rad(30.0)
    sub = 0.0
    gamma_s = 15e3
    gamma_w = 9.8e3

    soil_depth = mg.at_node["soil__depth"].copy()
    psi = sub * gamma_w * soil_depth
    slope = np.ones(mg.number_of_nodes) * slope_rad
    expected = ((cohesion_eff - psi * np.tan(phi)) / (gamma_s * soil_depth * np.sin(slope))) + (
        np.tan(phi) / np.tan(slope)
    )

    fos = factor_of_safety(
        mg, cohesion_eff, phi, submerged_soil_proportion=sub, soil_unit_weight=gamma_s, water_unit_weight=gamma_w
    )
    assert np.allclose(fos, expected, rtol=1e-6, atol=1e-9)


def test_critical_transient_acceleration_vectors(monkeypatch):
    mg = make_grid()
    slope_rad = np.deg2rad(20.0)
    monkeypatch.setattr(
        mg, "calc_slope_at_node", lambda elevs=None: np.ones(mg.number_of_nodes) * slope_rad
    )

    phi = np.deg2rad(30.0)
    coh = 500.0
    sub = 0.0
    gamma_s = 15e3
    gamma_w = 9.8e3
    g = 9.81

    a_h = np.ones(mg.number_of_nodes) * 0.3 * g
    a_v = np.ones(mg.number_of_nodes) * 0.1 * g

    h = mg.at_node["soil__depth"].copy()
    slope = np.ones(mg.number_of_nodes) * slope_rad
    psi = sub * gamma_w * h
    a_c_transient = (
        np.tan(phi) * (g * np.cos(slope) - a_v * np.cos(slope) - a_h * np.sin(slope))
        + ((g * coh) - (psi * g * np.tan(phi))) / (gamma_s * h)
        - g * np.sin(slope)
    )
    a_s_t = a_h * np.cos(slope) - a_v * np.sin(slope)
    a_c_transient[mg.boundary_nodes] = 0
    a_diff = a_s_t - a_c_transient

    ac, aslip, adiff = critical_transient_acceleration(
        mg, coh, phi, sub, a_h=a_h, a_v=a_v, soil_unit_weight=gamma_s, water_unit_weight=gamma_w, g=g
    )
    assert np.allclose(ac, a_c_transient, rtol=1e-6)
    assert np.allclose(aslip, a_s_t, rtol=1e-6)
    assert np.allclose(adiff, a_diff, rtol=1e-6)


def test_newmark_displacement_masks_unlabeled_and_scales_with_time():
    mg = make_grid()
    a_diff = np.zeros(mg.number_of_nodes)
    labels = np.zeros(mg.shape, dtype=int)
    labels[2:4, 2:4] = 1
    idx = np.where(labels.ravel() == 1)[0]
    a_diff[idx] = 2.0  # m/s^2

    disp = calculate_newmark_displacement(mg, a_diff, labels, time_shaking=3.0)
    # unlabeled nodes should be NaN
    unlabeled = np.where(labels.ravel() == 0)[0]
    assert np.all(np.isnan(disp[unlabeled]))
    # labeled nodes: s = 0.5 * a * t^2 = 0.5 * 2 * 9 = 9
    assert np.allclose(disp[idx], 9.0)
