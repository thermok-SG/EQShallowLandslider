"""
Functions for Newmark analysis
"""

# %% Required packages
import numpy as np

# %% Factor of Safety - Infinite slope

def factor_of_safety_inf_slope(grid,cohesion_eff,angle_int_frict,
                            submerged_soil_proportion,
                            soil_unit_weight=15e3,water_unit_weight=9.8e3
                            ):
    # Defined parameters
    # soil_unit_weight = 15e3 # N/m3 - typical top soil
    # water_unit_weight = 9.8e3 # N/m3 - specific weight of water at 4C
    
    soil_depth = np.array(grid["node"]["soil__depth"])
    slopes = np.array(grid.calc_slope_at_node(elevs='topographic__elevation'))
    
    slopes[slopes==0] += np.radians(1)
    soil_depth[soil_depth==0] += 0.001
    
    # Calculations
    x = cohesion_eff / (soil_unit_weight * soil_depth * np.sin(slopes))
    y = np.tan(np.radians(angle_int_frict))/np.tan(slopes)
    z = (submerged_soil_proportion * water_unit_weight * np.tan(np.radians(angle_int_frict)))/(soil_unit_weight * np.tan(slopes))
    
    factor_of_safety_inf_slope = x + y - z
    factor_of_safety_inf_slope[factor_of_safety_inf_slope>1e3] = np.nan
    
    return factor_of_safety_inf_slope
# %% Factor of safety

def factor_of_safety(grid,cohesion_eff,angle_int_frict,submerged_soil_proportion=0.5,
                    soil_unit_weight=15e3,water_unit_weight=9.8e3
                    ):
    
    soil_depth = np.array(grid["node"]["soil__depth"])
    slope = np.array(grid.calc_slope_at_node(elevs='topographic__elevation'))
    
    slope[slope==0] += np.nan
    soil_depth[soil_depth==0] += 0.001
    
    # Alternative to matric suction for saturated soils
    psi = submerged_soil_proportion*water_unit_weight*soil_depth
    
    #Eq. 4 Huang et al. 
    factor_of_safety = (
        (cohesion_eff - psi*np.tan(angle_int_frict))/(soil_unit_weight*soil_depth*np.sin(slope)) 
        + np.tan(angle_int_frict)/np.tan(slope)
        )
    
    return factor_of_safety

# %% Critical acceleration

def critical_acceleration(grid,cohesion_eff,angle_int_frict,submerged_soil_proportion,
                        soil_unit_weight=15e3,water_unit_weight=9.8e3,
                        g=9.81 # acceleration due to gravity; m/s2
                        ):
    
    """
    constant critical acceleration (a_c) = (factor_of_safety - 1) * g * sin(slope)
    """
    
    soil_depth = np.array(grid["node"]["soil__depth"])
    slope = np.array(grid.calc_slope_at_node(elevs='topographic__elevation'))
    
    slope[slope==0] += np.nan
    soil_depth[soil_depth==0] += 0.001
    
    # Alternative to matric suction for saturated soils
    psi = submerged_soil_proportion * water_unit_weight * soil_depth
    
    a_c =( 
        np.tan(angle_int_frict) * g * np.cos(slope)
        +((g*cohesion_eff)-(psi*g*np.tan(angle_int_frict)))/(soil_unit_weight*soil_depth) 
        - g*np.sin(slope)
        )
    
    return a_c

# %%% Critical transient acceleration

def critical_transient_acceleration(grid,cohesion_eff,angle_int_frict,
                                    submerged_soil_proportion,a_h=0,a_v=0,
                                    soil_unit_weight=15e3,water_unit_weight=9.8e3,
                                    g=9.81 # acceleration due to gravity; m/s2
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
    soil_depth[soil_depth==0] += 0.001 # Avoids division by zero
    
    slope = np.array(grid.calc_slope_at_node(elevs='topographic__elevation'))
    
    if submerged_soil_proportion >= 0:
        # Alternative to matric suction for saturated soils
        psi = submerged_soil_proportion * water_unit_weight * soil_depth
    elif submerged_soil_proportion < 0:
        psi = -15e3 # average matric suction based on Huang et al. 2020 and GEO report 1998
    
    # critical transient acceleration (a_c_transient) in 3D
    a_c_transient = ( 
        np.tan(angle_int_frict) * (g*np.cos(slope) - a_v*np.cos(slope) - a_h*np.sin(slope)) + 
        ((g*cohesion_eff)-(psi*g*np.tan(angle_int_frict)))/(soil_unit_weight*soil_depth) 
        - g*np.sin(slope)
        )
    
    # Driving acceleration downslope
    a_s_t = a_h * np.cos(slope) - a_v * np.sin(slope)
    
    a_c_transient[grid.boundary_nodes] = 0
    
    a_difference = a_s_t - a_c_transient
    
    return a_c_transient, a_s_t, a_difference

# %% Newmark displacement

def calculate_newmark_displacement(grid, a_difference, filtered_labeled_array, 
                                   time_shaking=0,
                                   ):
    """
    Calculates the total displacement per time segment.
    Could also be used in a loop to integrate the displacement over time.
    
    Parameters
    ----------
    grid : Landlab grid object
        landlab grid
    a_difference : Array of float64
        Difference between sliding and critical transient acceleration
        Output array from critical_transient_acceleration
    filtered_labeled_array : Array of int32
        Filtered unstable regions output from filter_regions_by_aspect 
    time_shaking : float, optional
        Time over which the excess sliding acceleration applied. The default is 0.

    Returns
    -------
    newmark_displacement : Array of float64
        Maximum displacement moved during excess shaking

    """
    
    a_diff = a_difference.reshape(grid.shape)

    filtered_regions = filtered_labeled_array == 0
    a_diff[filtered_regions] = np.nan

    newmark_displacement = 0.5 * a_diff * time_shaking ** 2
    
    return newmark_displacement.flatten()