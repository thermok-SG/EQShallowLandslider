
# %% Import required packages
import numpy as np

# %% Newmark displacement

def calculate_newmark_displacement(
    grid,
    a_difference,
    filtered_labeled_array,
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

    newmark_displacement = 0.5 * a_diff * time_shaking**2

    return newmark_displacement.flatten()