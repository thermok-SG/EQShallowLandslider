# %% List of major functions used elsewhere
# %% Required components
from pathlib import Path

import numpy as np
import math
from tqdm import tqdm
import heapq
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from scipy.ndimage import (
    label, labeled_comprehension,
    gaussian_filter, binary_dilation,
    generate_binary_structure, center_of_mass
    )

from skimage.measure import regionprops

from rasterio.features import rasterize

from bmi_topography import Topography
from landlab.io import esri_ascii
from landlab import (
    RasterModelGrid, imshowhs_grid # to plot results
    )

# %% ### General function list ###
# %%% Calculate number of output files
def num_output_files(time_step: int, total_model_time: float, output_interval: int) -> int:
    """
    Calculates the number of output files to expect at the model run

    Parameters
    ----------
    time_step : int
        Number of years in each model timestep.
    total_model_time : float
        Total number of years that the model will be running.
    output_interval : int
        Number of years making up one output cycle.

    Returns
    -------
    num_output_files : int
        Number of output files to expect at the end of the model.

    """
    
    # function to calculate Least Common Multiple
    def lcm(a,b):
        a = int(a)
        b = int(b)
        return abs(a * b) // math.gcd(a, b)
    
    # Calculate LCM of the time_step and the output_interval
    lcm_value = lcm(time_step, output_interval)
    
    # Calculate the number of output_files
    num_output_files = total_model_time // lcm_value
    
    return num_output_files
    


# %%% Plotting imported topography
def plotting_hy(grid, topo=True, DA=True, hill_DA=False, flow_metric="D8", hill_flow_metric="Quinn"):
    
    if topo:
        azdeg = 200
        altdeg = 20
        ve = 1
        plt.figure()
        plot_type = "DEM"
        ax = imshowhs_grid(
            grid,
            "topographic__elevation",
            grid_units=("deg", "deg"),
            var_name="Topo, m",
            cmap="terrain",
            plot_type=plot_type,
            vertical_exa=ve,
            azdeg=azdeg,
            altdeg=altdeg,
            default_fontsize=12,
            cbar_tick_size=10,
            cbar_width="100%",
            cbar_or="vertical",
            bbox_to_anchor=[1.03, 0.3, 0.075, 14],
            colorbar_label_y=-15,
            colorbar_label_x=0.5,
            ticks_km=False,
        )
        plt.title("DEM")
        
    if DA:
        ### Plot first instance of drainage_area
        grid.at_node["drainage_area"][grid.at_node["drainage_area"] == 0] = (
            grid.dx * grid.dx
        )
        plot_DA = np.log10(grid.at_node["drainage_area"] * 111e3 * 111e3)

        plt.figure()
        plot_type = "Drape1"
        drape1 = plot_DA
        thres_drape1 = None
        alpha = 0.5
        cmap1 = "terrain"
        ax = imshowhs_grid(
            grid,
            "topographic__elevation",
            grid_units=("deg", "deg"),
            cmap=cmap1,
            plot_type=plot_type,
            drape1=drape1,
            vertical_exa=ve,
            azdeg=azdeg,
            altdeg=altdeg,
            thres_drape1=thres_drape1,
            alpha=alpha,
            default_fontsize=12,
            cbar_tick_size=10,
            var_name="$log^{10}DA, m^2$",
            cbar_width="100%",
            cbar_or="vertical",
            bbox_to_anchor=[1.03, 0.3, 0.075, 14],
            colorbar_label_y=-15,
            colorbar_label_x=0.5,
            ticks_km=False,
        )

        plt.title("Drainage area")
        props = dict(boxstyle="round", facecolor="white", alpha=0.6)
        textstr = flow_metric
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

    if hill_DA:
        ### Plot second instance of drainage_area (hill_drainage_area)
        grid.at_node["hill_drainage_area"][grid.at_node["hill_drainage_area"] == 0] = (
            grid.dx * grid.dx
        )
        np.log10(grid.at_node["hill_drainage_area"] * 111e3 * 111e3)

        plt.figure()
        plot_type = "Drape1"
        # plot_type='Drape2'
        drape1 = np.log10(grid.at_node["hill_drainage_area"])
        thres_drape1 = None
        alpha = 0.5
        cmap1 = "terrain"
        ax = imshowhs_grid(
            grid,
            "topographic__elevation",
            grid_units=("deg", "deg"),
            cmap=cmap1,
            plot_type=plot_type,
            drape1=drape1,
            vertical_exa=ve,
            azdeg=azdeg,
            altdeg=altdeg,
            thres_drape1=thres_drape1,
            alpha=alpha,
            default_fontsize=10,
            cbar_tick_size=10,
            var_name="$log^{10}DA, m^2$",
            cbar_width="100%",
            cbar_or="vertical",
            bbox_to_anchor=[1.03, 0.3, 0.075, 14],
            colorbar_label_y=-15,
            colorbar_label_x=0.5,
            ticks_km=False,
        )

        plt.title("Hill drainage")
        props = dict(boxstyle="round", facecolor="white", alpha=0.6)
        textstr = hill_flow_metric
        ax.text(
            0.05,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )
        


# %%% Create mask from input shapefile
def create_shapefile_mask(grid, gdf):
   """
   Create a binary mask of shapefile polygons matching a raster's shape
   
   Args:
       gdf (GeoDataFrame): Shapefile geodataframe
       raster_shape (tuple): Shape of target raster (height, width)
       transform (affine.Affine): Raster's geotransform
   
   Returns:
       numpy.ndarray: Binary mask where 1 = polygon, 0 = outside
   """
   def create_transform(grid):
    """
    Extract geotransform from a Landlab grid
    
    Args:
        grid (landlab.grid): Landlab grid object
    
    Returns:
        affine.Affine: Geotransform for the grid
    """
    from affine import Affine

    # Assuming origin is at lower-left corner
    return Affine(grid.dx,   # pixel width
                  0,         # x rotation
                  grid.node_x.min(),  # x origin
                  0,         # y rotation
                  -grid.dy,  # pixel height (negative for top-down)
                  grid.node_y.max()  # y origin
                  )

   transform = create_transform(grid)

   mask = rasterize(
       gdf.geometry,
       out_shape=grid.shape,
       transform=transform,
       fill=0,
       default_value=1
   )
   return mask.astype(bool)
# %% ### Topographic functions ###
# %%% Gaussian curvature
def calculate_curvature_gaussian(elevation_grid):
    """
    Calculate the Gaussian curvature for an elevation grid stored as a numpy array.
    Might already exist in PriorityFloodFlowRouter (RichDEM)
    
    Parameters:
    elevation_grid (numpy.ndarray): 2D array representing the elevation grid.
    
    Returns:
    numpy.ndarray: 2D array representing the curvature.
    """
    # Calculate the gradients in the x and y directions
    grad_y, grad_x = np.gradient(elevation_grid)
    
    # Calculate the second-order gradients
    grad_yy, grad_yx = np.gradient(grad_y)
    grad_xy, grad_xx = np.gradient(grad_x)
    
    # Calculate the curvature
    curvature = (grad_xx * (1 + grad_y**2) - 2 * grad_x * grad_y * grad_xy + grad_yy * (1 + grad_x**2)) / ((1 + grad_x**2 + grad_y**2)**1.5)
    
    return curvature

# %%% Chi values
def calculate_chi(grid, drainage_area, theta=0.45):
    """
    Calculate chi values using Landlab's 1D node arrays and flow routing.
    
    Parameters:
    -----------
    grid : LandLabGrid
        Landlab grid with flow routing
    drainage_area : ndarray
        1D array of drainage area at each node
    theta : float
        Concavity index (typically 0.45)
    
    Returns:
    --------
    chi : ndarray
        1D array of chi values for each node
    """
    # Get flow routing fields
    flow_receiver = grid.at_node['flow__receiver_node']
    
    # Initialize chi array
    chi = np.zeros(grid.number_of_nodes)
    
    # Reference area (A0)
    A0 = np.min(drainage_area[drainage_area > 0])
    
    # Work upstream from outlets using flow receivers
    # Start with core nodes (non-boundary)
    core_nodes = grid.core_nodes
    
    # Calculate chi by integrating upstream
    for node in core_nodes:
        current_node = node
        next_node = flow_receiver[current_node]
        
        while current_node != next_node:  # while not at outlet/pit
            # Calculate dx using grid spacing
            dx = grid.dx
            chi[current_node] = chi[next_node] + (A0/drainage_area[current_node])**theta * dx
            current_node = next_node
            next_node = flow_receiver[current_node]
            
    return chi

# %%% Channel steepness
def calculate_channel_steepness(grid, chi, elevs, channel_nodes):
    """
    Calculate channel steepness using Landlab's 1D node arrays.
    
    Parameters:
    -----------
    grid : LandLabGrid
        Landlab grid
    chi : ndarray
        1D array of chi values
    elevs : ndarray
        1D array of elevation values
    channel_nodes : ndarray
        1D array of channel node IDs
    
    Returns:
    --------
    ksn : ndarray
        1D array of channel steepness values
    """
    ksn = np.zeros(grid.number_of_nodes)
    
    # Get flow receiver information
    flow_receiver = grid.at_node['flow__receiver_node']
    
    # Calculate ksn for channel nodes
    for node in channel_nodes:
        receiver = flow_receiver[node]
        if receiver != node:  # if not a sink
            dchi = chi[node] - chi[receiver]
            if dchi != 0:
                ksn[node] = (elevs[node] - elevs[receiver]) / dchi
    
    return ksn

# %%% Excess topography
# %%%% Non-iterative
def calculate_excess_topography(grid, drainage_area, slopes, theta=0.45,
                                    channel_nodes=None, slope_threshold=None):
    """
    Calculate excess topography using chi-derived reference surface with slope filtering.
    
    Parameters:
    -----------
    grid : LandLabGrid
        Landlab grid with flow routing
    drainage_area : ndarray
        1D array of drainage area values
    slopes : ndarray
        1D array of precomputed slopes at each node
    channel_nodes : ndarray, optional
        1D array of channel node IDs
    theta : float
        Concavity index (default is 0.45)
    slope_threshold : float, optional
        Threshold slope for filtering nodes (default: None, no filtering)
    
    Returns:
    --------
    excess_topo : ndarray
        1D array of excess topography values
    reference_surface : ndarray
        1D array of reference surface elevations based on chi
    """
    
    slope_threshold = np.radians(slope_threshold)
    # Get elevation data
    elevs = grid.at_node['topographic__elevation']
    valid_mask = ~np.isnan(elevs)
    
    # Calculate chi values
    chi = calculate_chi(grid, drainage_area, theta)
    
    # Identify channels if not provided
    if channel_nodes is None:
        threshold = np.percentile(drainage_area[grid.core_nodes], 90)
        channel_nodes = grid.core_nodes[drainage_area[grid.core_nodes] > threshold]
    
    # Filter channel nodes by slope threshold if provided
    if slope_threshold is not None:
        channel_nodes = channel_nodes[slopes[channel_nodes] <= slope_threshold]
    
    # Fit a linear relationship between chi and elevation for filtered channel nodes
    channel_chi = chi[channel_nodes]
    channel_elevs = elevs[channel_nodes]
    coeffs = np.polyfit(channel_chi, channel_elevs, 1)
    ksn = coeffs[0]  # Slope of chi-elevation regression
    intercept = coeffs[1]  # Intercept of chi-elevation regression
    
    # Calculate reference surface using chi
    reference_surface = np.full(grid.number_of_nodes, np.nan)
    reference_surface[valid_mask] = chi[valid_mask] * ksn + intercept
    
    # Calculate excess topography
    excess_topo = np.full(grid.number_of_nodes, np.nan)
    excess_topo[valid_mask] = elevs[valid_mask] - reference_surface[valid_mask]
    excess_topo[valid_mask] = np.maximum(excess_topo[valid_mask], 0)
    
    return excess_topo, reference_surface

# %%% Aspect rose diagram
def plot_aspect_roses(datasets, labels=None, colors=None, normalize=True, log_scale=False):
    """
    Create a rose diagram with multiple normalized datasets.
    
    Parameters:
    datasets : list of array-like
        List of aspect value arrays
    labels : list of str, optional
        Labels for each dataset
    colors : list of str, optional
        Colors for each dataset
    normalize : bool, default=True
        If True, normalize each dataset as percentage of total
    log_scale : bool, default=False
        If True, use log10 scale for radial axis
    """
    # Ensure datasets are numpy arrays
    datasets = [np.array(dataset) % 360 for dataset in datasets]

    # Default labels and colors if not provided
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(datasets))]
    if colors is None:
        colors = ['skyblue', 'lightgreen', 'salmon', 'purple', 'orange']

    # Set up the plot
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, projection='polar')

    # Number of bins
    n_bins = 16
    max_percentage = 0

    # Plot each dataset
    for i, (dataset, label_num, color) in enumerate(zip(datasets, labels, colors)):
        # Create histogram
        hist, bin_edges = np.histogram(dataset, bins=n_bins, range=(0, 360))

        # Normalize if requested
        if normalize:
            hist = (hist / len(dataset)) * 100

        # Track maximum percentage for scaling
        max_percentage = max(max_percentage, np.max(hist))

        # Calculate bin centers (in radians)
        bin_centers = np.deg2rad(bin_edges[:-1] + np.diff(bin_edges)/2)

        # Width of each bar (in radians)
        width = np.deg2rad(360 / n_bins)

        # Apply log scale if requested
        if log_scale:
            # Add small constant to avoid log(0)
            hist = np.log10(hist + 1)

        # Plot the rose diagram with partial transparency
        ax.bar(bin_centers, hist, width=width, bottom=0.0,
               color=color, edgecolor='black', alpha=0.5,
               label=f'{label_num} (n={len(dataset)})')

    # Customize the plot
    ax.set_theta_zero_location('N')  # 0 degrees at the top
    ax.set_theta_direction(-1)  # Clockwise

    # Set title based on scale type
    scale_type = "Log10 " if log_scale else ""
    norm_type = "Normalized " if normalize else ""
    ax.set_title(f'{scale_type}{norm_type}Topographic Aspect Rose Diagram')

    # Add labels to show cardinal directions
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

    # Customize radial labels based on scale type
    if log_scale:
        ax.set_rticks([0, 0.5, 1, 1.5, 2])
        ax.set_rticklabels(['0', '0.5', '1', '1.5', '2'])
        plt.text(0, 2.2, 'log10(percentage + 1)', ha='center', va='bottom')
    # else:
    #     if normalize:
    #         plt.text(0, max_percentage * 1.1, 'Percentage of observations', ha='center', va='bottom')
    #     else:
    #         plt.text(0, max_percentage * 1.1, 'Count', ha='center', va='bottom')

    # Add legend
    plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0))

    plt.tight_layout()
    plt.show()









# %%%% Trace landslide paths with a slope threshold
def trace_paths_landslides_slopes(grid, starting_nodes, slope_array, slope_thresh=10,
                           check_sum=False, print_path_details=False):
    """
    Calculates all downslope paths taken from a set of starting nodes until 
    reaching nodes with slopes below the threshold

    Parameters
    ----------
    grid : Landlab RasterModelGrid
        DESCRIPTION.
    starting_nodes : numpy.ndarray
        1D array of node ids to trace paths for.
    slope_array : numpy.ndarray
        Array containing slope values for each node
    slope_thresh : float, optional
        Slope threshold in degrees. The default is 10.
    check_sum : bool, optional
        Checks the sum of final proportions for all paths from each node.
        The sum should equal to one. Prints node_ids that do not add up to one
        The default is False.
    print_path_details : bool, optional
        Lists each starting node_id and all subsequent node_ids and proportions 
        for the paths starting at that node.
        The default is False.

    Returns
    -------
    final_nodes: list of tuples
        Lists each path as a tuple of node_ids, starting with the initial node.
    final_proportions: list
        List containing the final proportions for each path defined in final_nodes
    path_details: dict of lists
        Dictionary of starting node_ids with a list of paths for each node
    
    if check_sum=True:
        proportion_not_zero: dict
            Dictionary listing each path where the final proportions do not 
            add up to one
    """
    slope_thresh_rad = np.radians(slope_thresh)
    receiver_nodes = grid.at_node['hill_flow__receiver_node']
    receiver_proportions = grid.at_node['hill_flow__receiver_proportions']
    boundary_nodes = set(grid.boundary_nodes)  # Convert to set for faster lookup
    
    final_nodes = []
    final_proportions = []
    path_details = {}  # To store details of each path for each starting node
    
    for node in tqdm(starting_nodes):
        stack = [(node, 1.0, [node])]  # (current_node, current_proportion, path)
        path_details[node] = []  # Initialize path details for this starting node

        while stack:
            current_node, current_proportion, path = stack.pop()  # Using pop() instead of heapq since we don't need distance ordering

            # Check if current node is below slope threshold
            if slope_array[current_node] <= slope_thresh_rad:
                final_nodes.append((node, current_node))
                final_proportions.append(current_proportion)
                path_details[node].append((path, current_proportion))
                continue
            
            receivers = receiver_nodes[current_node]
            receiver_props = receiver_proportions[current_node]
            
            if receivers[0] == current_node:
                # If no receivers but slope still high, store current location
                final_nodes.append((node, current_node))
                final_proportions.append(current_proportion)
                path_details[node].append((path, current_proportion))
                continue
            
            for receiver, prop in zip(receivers, receiver_props):
                if receiver != -1:
                    new_path = path + [receiver]
                    new_proportion = current_proportion * prop
    
                    # Check if the receiver node is a boundary node
                    if receiver in boundary_nodes or grid.at_node['topographic__elevation'][receiver] == np.nan:
                        final_nodes.append((node, current_node))
                        final_proportions.append(current_proportion)
                        path_details[node].append((path, current_proportion))
                    else:
                        stack.append((receiver, new_proportion, new_path))
    
    # Prints out all of the path node ids
    if print_path_details:
        print("Path Details:")
        for start_node in path_details:
            print(f"Starting Node: {start_node}")
            for path, proportion in path_details[start_node]:
                print(f"Path: {path}, Proportion: {proportion}")
    
    # Checks the sum total for all proportions of the paths
    if check_sum:                        
        proportion_sums = {}
        proportion_not_zero = []
        for (start_node, end_node), proportion in zip(final_nodes, final_proportions):
            if start_node not in proportion_sums:
                proportion_sums[start_node] = 0.0
            proportion_sums[start_node] += proportion
        for start_node, total_proportion in proportion_sums.items():
            if not np.isclose(total_proportion, 1.0):
                proportion_not_zero.append([start_node, total_proportion])
                
        return final_nodes, final_proportions, path_details, proportion_not_zero
    
    else:
        return final_nodes, final_proportions, path_details
    



