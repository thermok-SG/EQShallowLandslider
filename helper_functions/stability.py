"""
Slope stability calculation functions for ShallowLandslider

Author: sghoshal
"""
import numpy as np

def factor_of_safety(
    grid,
    cohesion_eff,
    angle_int_frict,
    submerged_soil_proportion=0.5,
    soil_unit_weight=15e3,
    water_unit_weight=9.8e3,
):
    soil_depth = np.array(grid["node"]["soil__depth"])
    slope = np.array(grid.calc_slope_at_node(elevs="topographic__elevation"))

    slope[slope == 0] += np.nan
    soil_depth[soil_depth == 0] += 0.001

    # Alternative to matric suction for saturated soils
    psi = submerged_soil_proportion * water_unit_weight * soil_depth

    # Eq. 4 Huang et al.
    factor_of_safety = (cohesion_eff - psi * np.tan(angle_int_frict)) / (
        soil_unit_weight * soil_depth * np.sin(slope)
    ) + np.tan(angle_int_frict) / np.tan(slope)

    return factor_of_safety

# %% Critical transient acceleration calculator

def critical_transient_acceleration(
    grid,
    cohesion_eff,
    angle_int_frict,
    submerged_soil_proportion,
    a_h=0,
    a_v=0,
    soil_unit_weight=15e3,
    water_unit_weight=9.8e3,
    g=9.81,  # acceleration due to gravity; m/s2
):
    """
    Calculates the sliding and critical acceleration for each node of the grid
    Also calculates the difference in accelerations

    Parameters
    ----------
    grid : Landlab grid object
        landlab grid
    cohesion_eff : float
        Effective cohesion. Specified in pascals. Can also be array of floats
    angle_int_frict : float64
        Angle of internal friction in radians
    submerged_soil_proportion : float
        Proportion of soil that is submerged
    a_h : float, optional
        Horizontal PGA component. Generally a float value times 'g'.
        Can also be an array. The default is 0.
    a_v : float, optional
        Vertical PGA component. Generally a float value times 'g'.
        Can also be an array. The default is 0.
    soil_unit_weight : float, optional
        Unit weight of soil in N/m^3. The default is 15e3.
    water_unit_weight : float, optional
        Unit weight of water in N/m^3. The default is 9.8e3.
    g : float, optional
        Gravitational acceleration in m/s^2. The default is 9.81.

    Returns
    -------
    a_c_transient : array of float64
        Critical transient acceleration. Outputs as a flattened 1-D array
    a_s_t : array of float64
        Seismically-triggered sliding acceleration. Outputs as a flattened 1-D array.
    a_difference : array of float64
        Difference in accelerations. Outputs as a flattened 1-D array.

    """

    soil_depth = np.array(grid["node"]["soil__depth"])
    soil_depth[soil_depth == 0] += 0.001  # Avoids division by zero

    slope = np.array(grid.calc_slope_at_node(elevs="topographic__elevation"))

    if submerged_soil_proportion >= 0:
        # Alternative to matric suction for saturated soils
        psi = submerged_soil_proportion * water_unit_weight * soil_depth
    elif submerged_soil_proportion < 0:
        psi = (
            -15e3
        )  # average matric suction based on Huang et al. 2020 and GEO report 1998

    # critical transient acceleration (a_c_transient) in 3D
    a_c_transient = (
        np.tan(angle_int_frict)
        * (g * np.cos(slope) - a_v * np.cos(slope) - a_h * np.sin(slope))
        + ((g * cohesion_eff) - (psi * g * np.tan(angle_int_frict)))
        / (soil_unit_weight * soil_depth)
        - g * np.sin(slope)
    )

    # Driving acceleration downslope
    a_s_t = a_h * np.cos(slope) - a_v * np.sin(slope)

    a_c_transient[grid.boundary_nodes] = 0

    a_difference = a_s_t - a_c_transient

    return a_c_transient, a_s_t, a_difference