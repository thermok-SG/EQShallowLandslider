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
    
# %%% Getting topography from OpenTopography
def get_topo(buffer: float, north=28.25, south=28.23, east=85.18, west=85.15, dem_type="NASADEM"):
    """
    Downloads DEM from OpenTopo and generates a landlab RasterModelGrid.

    Parameters
    ----------
    buffer : float
        Additional space around the DEM to remove potential edge effects.
        Values in decimal degrees.
    north : float, optional
        Northern extent of DEM in decimal degrees. The default is 28.25.
    south : float, optional
        Southern extent of DEM in decimal degrees. The default is 28.23.
    east : float, optional
        Eastern extent of DEM in decimal degrees. The default is 85.18.
    west : float, optional
        Western extent of DEM in decimal degrees. The default is 85.15.
    dem_type : string, optional
        Type of DEM to download from OpenTopo. The default is "NASADEM".
        Available DEM types:
            1. SRTMGL3: Default, 90 m
            2. SRTMGL1: 30 m
            3. AW3D30: ALOS World 3D, 30 m
            4. NASADEM: NASADEM Global DEM, 30 m
            5. COP30: Copernicus Global DSM 30 m
            6. COP90: Copernicus Global DSM 90 m

    Returns
    -------
    grid : RasterModelGrid
        Elevation grid.
    z_geog : ndarray
        Array of elevation values.

    """

    params = Topography.DEFAULT.copy()
    params["south"] = south - buffer
    params["north"] = north + buffer
    params["west"] = west - buffer
    params["east"] = east + buffer
    params["dem_type"] = dem_type
    params["output_format"] = "AAIGrid"
    params["cache_dir"] = Path.cwd()
    params["api_key"] = "f08b2664772eb044626d5cb114924de1"
    dem = Topography(**params)
    name = dem.fetch()
    props = dem.load()
    # dim_x = props.sizes["x"]
    # dim_y = props.sizes["y"]
    # cells = props.sizes["x"] * props.sizes["y"]
    
    with open(name) as fp:
        grid_geog = esri_ascii.load(fp, name="topographic__elevation", at="node")
    
    z_geog = grid_geog.at_node["topographic__elevation"]
    
    match dem_type:
        case "SRTMGL3" | "COP90":
            grid_spacing = 90
        case "SRTMGL1" | "AW3D30" | "NASADEM" | "COP30":
            grid_spacing = 30
    
    grid = RasterModelGrid(
        (grid_geog.number_of_node_rows, grid_geog.number_of_node_columns),
        xy_spacing=grid_spacing,
        xy_axis_units='m',
    )
    grid.add_field("topographic__elevation",
                   z_geog, at="node")

    print(params)
    print(props)
    
    return grid, z_geog

# %%% Smoothen landlab grid
def smooth_elevation_grid(grid, method='mean', smooth_num=1):
    """
    Apply a 3x3 moving window smoothing to a landlab grid's elevation data.
    
    Parameters:
    -----------
    grid : landlab.grid
        The landlab grid containing elevation data
    method : str, optional
        The smoothing method to use: 'mean' or 'gaussian' (default: 'mean')
        
    Returns:
    --------
    numpy.ndarray
        The smoothed elevation array
    """
    if method == 'mean':
        from scipy.ndimage import uniform_filter
    elif method == 'gaussian':
        # For Gaussian smoothing
        from scipy.ndimage import gaussian_filter
    else:
        raise ValueError("Method must be 'mean' or 'gaussian'")
    
    # Get the original elevation data
    elevation = grid.at_node['topographic__elevation'].copy()
    
    # Reshape to 2D for smoothing (assuming a raster grid)
    elevation_2d = elevation.reshape(grid.shape)
    
    smooth_round = 0
    while smooth_round < smooth_num:
    
        if method == 'mean':
            # Apply a 3x3 mean filter
            smoothed_elevation = uniform_filter(elevation_2d, size=3, mode='reflect')
        elif method == 'gaussian':
            # For Gaussian smoothing
            smoothed_elevation = gaussian_filter(elevation_2d, sigma=1, mode='reflect')
        smooth_round += 1
    
    # Reshape back to 1D for use with landlab
    smoothed_elevation_1d = smoothed_elevation.flatten()
    
    return smoothed_elevation_1d

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
        
# %%% Distance between nodes
def calculate_node_distance(grid, node1, node2):
    """
    Calculates the rectilinear distance between two nodes in the grid

    Parameters
    ----------
    grid : Landlab RasterModelGrid
        DESCRIPTION.
    node1 : int
        Id of first node.
    node2 : int
        Id of second node.

    Returns
    -------
    float
        Rectilinear distance between two given nodes.

    """
    x1, y1 = grid.node_x[node1], grid.node_y[node1]
    x2, y2 = grid.node_x[node2], grid.node_y[node2]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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

# %% ### Region creation functions ###
# %%% Proximity weighting function
'''
Calculates a grid of weights based on the proximity of TRUE cells to each other
'''
def proximity_weighting(binary_grid):
    center = np.array(binary_grid.shape) / 2
    indices = np.indices(binary_grid.shape)
    distances = np.sqrt((indices[0] - center[0])**2 + (indices[1] - center[1])**2)
    max_distance = np.max(distances)
    return 1 - (distances / max_distance)

# %%% Density weighting function

def gaussian_density_weighting(binary_grid):
    return gaussian_filter(binary_grid.astype(float), sigma=1)

# %%% Calculate regions from binary grid
def calculate_regions(binary_grid, proximity_weight_on=False, density_weight_on=False,
                      proximity_weight_val=0, density_weight_val=0,
                      threshold_val=0, connect_val=4
                      ):
    """
    Separates disconnected areas from each other and labels them as individual regions

    Parameters
    ----------
    binary_grid : array of float64 (to be deprecated)
        pseudo-boolean array showing regions of instability.
    proximity_weight_on : bool, optional
        Select whether proximity weighting is on. The default is False.
    density_weight_on : bool, optional
        Select whether gaussian density weighting is on. The default is False.
    proximity_weight_val : float, optional
        Proximity weighting to be applied. The default is 0.
    density_weight_val : float, optional
        Density weighting to be applied. The default is 0.
    threshold_val : float, optional
        Minimum weight to apply to selecting regions. The default is 0.
    connect_val : int, optional
        Can be either 4 or 8. Defines the type of connectivity used to select regions.
        The default is 4.

    Raises
    ------
    ValueError
        1. Proximity and density weights need to add up to 1
        2. Only 4- and 8-connectivity possible

    Returns
    -------
    labeled_array : array of int32
        2D array of region labels.
    num_features : int
        Number of regions selected.
    
    TODO: Change this so that it takes in sliding and critical acceleration rather than separate sliding_array

    """
    
    # Calculate individual weights
    if proximity_weight_on:
        proximity_weights = proximity_weighting(binary_grid)
        proximity_weights /= np.max(proximity_weights) # Normalise weights
        
    if density_weight_on:
        density_weights = gaussian_density_weighting(binary_grid)
        density_weights /= np.max(density_weights) # Normalise weights

    # Combine the weights
    if proximity_weight_on and density_weight_on:
        if (proximity_weight_val + density_weight_val) != 1:
            raise ValueError('The weights need to add up to 1')
        else:
            combined_weights = (proximity_weight_val * proximity_weights) + (
                density_weight_val * density_weights)
            # Threshold the combined weights to create a binary grid
            weighted_binary_grid = combined_weights > threshold_val
    elif proximity_weight_on and not density_weight_on:
        weighted_binary_grid = proximity_weights > threshold_val
    elif density_weight_on and not proximity_weight_on:
        weighted_binary_grid = density_weights > threshold_val
    else:
        weighted_binary_grid = binary_grid > 0
    
    # Create binary array with 4- or 8-connectivity
    match connect_val:
        case 4:
            struct_arr = generate_binary_structure(2, 1)
        case 8:
            struct_arr = generate_binary_structure(2, 2)
        case _:
            raise ValueError('Only 4- or 8-connectivity possible')

    # Label distinct regions in the weighted binary grid
    labeled_array, num_features = label(weighted_binary_grid,
                                        structure=struct_arr)
    
    return labeled_array, num_features

# %%% Classify DEM by aspect

def create_zones(interval=20):
    """
    Create aspect zones with specified degree intervals.
    
    Parameters:
    interval : float
        Size of each zone in degrees
    
    Returns:
    dict
        Dictionary of zone ranges
    """
    zones = {}
    start = 0
    while start < 360:
        end = (start + interval) % 360
        zone_name = f"{start:03d}-{end:03d}"
        zones[zone_name] = (start, end)
        start += interval
    return zones

def zone_aspects(aspect_array, zones=None):
    """
    Categorize a 2D aspect array into defined zones.
    
    Parameters:
    aspect_array : numpy.ndarray
        2D array of aspect values in degrees (0-360)
    zones : dict, optional
        Dictionary mapping zone names to (min_angle, max_angle) tuples.
        If None, uses 20-degree intervals.
    
    Returns:
    numpy.ndarray
        2D array with integer zone labels
    """
    if zones is None:
        zones = create_zones(20)
    
    aspect_array = np.array(aspect_array, dtype=float)
    aspect_array = aspect_array % 360
    zone_array = np.full(aspect_array.shape, -1, dtype=int)
    
    for zone_idx, (_, (min_aspect, max_aspect)) in enumerate(zones.items()):
        if min_aspect > max_aspect:  # Handles wrap-around zones
            mask = ((aspect_array >= min_aspect) | (aspect_array <= max_aspect))
        else:
            mask = ((aspect_array >= min_aspect) & (aspect_array < max_aspect))
        zone_array[mask] = zone_idx
    
    return zone_array

def split_groups_by_aspect(groups, aspect_array, zones=None, 
                           min_size=2, handle_small='merge'):
    """
    Split connected component groups based on aspect zones.
    Each group that spans multiple aspect zones will be split into separate groups.
    Small regions can be filtered out or merged with neighbors.
    
    Parameters:
    -----------
    groups : numpy.ndarray
        2D array of integer labels where each connected component has a unique label
    aspect_array : numpy.ndarray
        2D array of aspect values in degrees (0-360)
    zones : dict, optional
        Dictionary mapping zone names to (min_angle, max_angle) tuples
    min_size : int, optional
        Minimum number of pixels for a region to be kept (default: 2)
    handle_small : str, optional
        How to handle regions smaller than min_size: 'merge' or 'remove' (default: 'merge')
        
    Returns:
    --------
    numpy.ndarray
        New array where groups have been split based on aspect zones
    numpy.ndarray
        Zone labels array
    dict
        Mapping of new group labels to (original_group, zone_name) pairs
    """
    
    # Get zone labels for aspects
    if zones is None:
        zones = create_zones(20)
    zone_labels = zone_aspects(aspect_array, zones)
    
    # Initialize output array and group info dictionary
    new_groups = np.zeros_like(groups)
    group_info = {}
    next_label = 1
    
    # Get list of zone names for lookup
    zone_names = list(zones.keys())
    
    # Store small regions for later processing
    small_regions = []
    
    # Process each original group
    pbar = tqdm(np.unique(groups))
    for group_id in pbar:
        if group_id == 0:  # Skip background
            continue
        
        pbar.set_description(f"Processing group number {group_id}")
        
        # Get mask for current group
        group_mask = (groups == group_id)
        
        # For each zone present in this group
        for zone_id in np.unique(zone_labels[group_mask]):
            # Create mask for this group in this zone
            zone_mask = (zone_labels == zone_id)
            combined_mask = group_mask & zone_mask
            
            # Label connected components within this zone
            labels, num_features = label(combined_mask)
            
            # Assign new unique labels to each component
            for label_name in range(1, num_features + 1):
                component_mask = (labels == label_name)
                component_size = np.sum(component_mask)
                
                if component_size < min_size:
                    # Store small region info for later processing
                    small_regions.append({
                        'mask': component_mask,
                        'group_id': group_id,
                        'zone_id': zone_id,
                        'size': component_size,
                        'centroid': center_of_mass(component_mask)
                    })
                else:
                    # This is a large enough region, so assign it a label
                    new_groups[component_mask] = next_label
                    group_info[next_label] = (group_id, zone_names[zone_id])
                    next_label += 1
    
    # Handle small regions based on chosen strategy
    if small_regions and handle_small == 'remove':
        # Small regions already excluded (do nothing)
        pass
    
    elif small_regions and handle_small == 'merge':
        # Process small regions in order from smallest to largest
        small_regions.sort(key=lambda x: x['size'])
        
        for region in small_regions:
            # Dilate the small region mask to find neighbors
            dilated = binary_dilation(region['mask'])
            neighbor_mask = dilated & ~region['mask']
            
            # Find neighboring regions in the new groups
            neighbor_labels = np.unique(new_groups[neighbor_mask])
            neighbor_labels = neighbor_labels[neighbor_labels > 0]  # Remove background
            
            if len(neighbor_labels) > 0:
                # Find the most common neighbor
                neighbor_counts = [(nl, np.sum(new_groups[neighbor_mask] == nl)) 
                                  for nl in neighbor_labels]
                best_neighbor = max(neighbor_counts, key=lambda x: x[1])[0]
                
                # Merge with best neighbor
                new_groups[region['mask']] = best_neighbor
            else:
                # No neighbors found, create a new region
                new_groups[region['mask']] = next_label
                group_info[next_label] = (region['group_id'], zone_names[region['zone_id']])
                next_label += 1
    
    return new_groups, zone_labels, group_info

# %% ### Region selection functions ###
# %%% Method 1
# TODO: add method for simple proportional selection of regions
# %%% Method 2: Select groups/proportion based on a_c
# %%%% Helper functions
def _calculate_acceleration_probability(pga_ratio, critical_acceleration):
    """
    Calculate landslide failure probability based on PGA ratio and critical acceleration.
    
    Parameters:
    -----------
    pga_ratio : float
        Ratio of Peak Ground Acceleration to critical acceleration
    critical_acceleration : float
        Threshold acceleration for slope failure
    
    Returns:
    --------
    float
        Probability of slope failure
    """
    # Base probability increases as critical acceleration decreases
    base_probability = 1 - np.exp(-5 * (1 / critical_acceleration))
    
    # PGA ratio effect
    pga_effect = 1 / (1 + np.exp(-5 * (pga_ratio - 1)))
    
    # Combined probability
    combined_probability = base_probability * pga_effect
    
    return np.clip(combined_probability, 0, 1)

def _calculate_slope_stability_factor(slope):
    """
    Calculate slope stability factor based on slope angle.
    
    Parameters:
    -----------
    slope : float
        Slope angle in degrees
    
    Returns:
    --------
    float
        Slope stability multiplier
    """
    if slope <= 30:
        return 1 + 0.02 * slope  # Gradual increase
    elif 30 < slope <= 45:
        return 1 + 0.6 * ((slope - 30) / 15)  # Accelerated increase
    else:
        return 2  # Maximum factor for very steep slopes

# %%%% Main functions
# %%%%% Group probability calculation
"""
Input Data → calculate_group_probability (for each region) → group_probs dictionary
↓
[If normalizing] group_probs → normalize_group_probabilities → normalized_group_probs
↓
group_probs or normalized_group_probs → apply_probabilities_to_array → probability_array
↓
All data → create_metadata → metadata
↓
Return (probability_array, metadata)
"""
# Calculates probabilities for each group
def calculate_group_probability(h_pga_grid, v_pga_grid, critical_acceleration_grid, 
                               mask, slope_grid=None, soil_array=None, 
                               geological_factor_array=None):
    """
    Calculate the landslide probability for a specific group/region.
    
    Parameters:
    -----------
    h_pga_grid : ndarray
        Horizontal Peak Ground Acceleration grid
    v_pga_grid : ndarray
        Vertical Peak Ground Acceleration grid
    critical_acceleration_grid : ndarray
        Critical acceleration thresholds grid
    mask : ndarray of bool
        Boolean mask identifying the region to analyze
    slope_grid : ndarray, optional
        Slope angles in degrees
    soil_array : ndarray, optional
        Soil susceptibility index
    geological_factor_array : ndarray, optional
        Geological instability factors
    
    Returns:
    --------
    dict
        Dictionary containing probability and factors for the group
    """
    # Local critical acceleration
    local_critical_acceleration = np.mean(critical_acceleration_grid[mask])
    
    # PGA Calculations
    group_h_pga = h_pga_grid[mask]
    group_v_pga = v_pga_grid[mask]
    
    # Vector resultant
    resultant_pga = np.sqrt(np.mean(group_h_pga)**2 + np.mean(group_v_pga)**2)
    
    # Acceleration Ratio Calculation
    pga_ratio = resultant_pga / local_critical_acceleration
    
    # Base Probability Model with Critical Acceleration
    base_prob = _calculate_acceleration_probability(pga_ratio, local_critical_acceleration)
    
    # Slope Factor
    if slope_grid is not None:
        mean_slope = np.mean(slope_grid[mask])
        slope_factor = _calculate_slope_stability_factor(mean_slope)
        base_prob *= slope_factor
    
    # Soil Condition Factor
    if soil_array is not None:
        soil_susceptibility = np.mean(soil_array[mask])
        soil_factor = 1 + soil_susceptibility
        base_prob *= soil_factor
    
    # Geological Factor
    if geological_factor_array is not None:
        geo_factor = np.mean(geological_factor_array[mask])
        base_prob *= (1 + 0.5 * geo_factor)
    
    # Stochastic Variability
    stochastic_factor = np.random.lognormal(mean=0, sigma=0.2)
    group_prob = np.clip(base_prob * stochastic_factor, 0, 1)
    
    return {
        'probability': group_prob,
        'critical_acceleration': local_critical_acceleration,
        'resultant_pga': resultant_pga,
        'pga_ratio': pga_ratio,
        'base_probability': base_prob
    }

# Normalise probabilities for groups 
def normalize_group_probabilities(group_probs):
    """
    Normalize probabilities across groups using min-max scaling.
    
    Parameters:
    -----------
    group_probs : dict
        Dictionary of group probabilities and metadata
    
    Returns:
    --------
    tuple
        (normalized_group_probs, normalization_metadata)
    """
    # Extract probabilities for all groups
    probs = [info['probability'] for info in group_probs.values()]
    
    if len(probs) <= 1:
        # Nothing to normalize with only one group
        norm_metadata = {
            'performed': False,
            'reason': 'Only one group present'
        }
        return group_probs, norm_metadata
    
    min_prob = min(probs)
    max_prob = max(probs)
    
    if max_prob <= min_prob:
        # No range to normalize
        norm_metadata = {
            'performed': False,
            'reason': 'All groups have the same probability',
            'value': min_prob
        }
        return group_probs, norm_metadata
    
    # Normalize each group's probability
    normalized_group_probs = {}
    for label_num, info in group_probs.items():
        # Copy the original info
        normalized_info = info.copy()
        
        # Apply min-max normalization
        normalized_prob = (info['probability'] - min_prob) / (max_prob - min_prob)
        normalized_info['normalized_probability'] = normalized_prob
        
        normalized_group_probs[label_num] = normalized_info
    
    norm_metadata = {
        'performed': True,
        'min_raw_prob': min_prob,
        'max_raw_prob': max_prob
    }
    
    return normalized_group_probs, norm_metadata

# Generate probability array with all groups
def apply_probabilities_to_array(probability_array, group_probs, normalized=False):
    """
    Apply calculated probabilities to the output array.
    
    Parameters:
    -----------
    probability_array : ndarray
        Output array to populate with probabilities
    group_probs : dict
        Dictionary of group probabilities and metadata
    normalized : bool
        Whether to use normalized probabilities
    
    Returns:
    --------
    ndarray
        Updated probability array
    """
    for label_num, info in group_probs.items():
        if normalized and 'normalized_probability' in info:
            probability_array[info['mask']] = info['normalized_probability']
        else:
            probability_array[info['mask']] = info['probability']
    
    return probability_array

# Create final metadata
def create_metadata(group_probs, probability_array, normalization_metadata, normalized=False):
    """
    Create metadata dictionary from analysis results.
    
    Parameters:
    -----------
    group_probs : dict
        Dictionary of group probabilities and metadata
    probability_array : ndarray
        Array of calculated probabilities
    normalization_metadata : dict
        Metadata about normalization process
    normalized : bool
        Whether normalization was performed
    
    Returns:
    --------
    dict
        Comprehensive metadata about the analysis
    """
    metadata = {'group_details': [], 'normalization': normalization_metadata}
    
    for label_num, info in group_probs.items():
        group_meta = {
            'label': label_num,
            'critical_acceleration': info['critical_acceleration'],
            'resultant_pga': info['resultant_pga'],
            'pga_ratio': info['pga_ratio'],
            'base_probability': info['base_probability'],
        }
        
        if normalized and 'normalized_probability' in info:
            group_meta['raw_probability'] = info['probability']
            group_meta['final_probability'] = info['normalized_probability']
        else:
            group_meta['final_probability'] = info['probability']
        
        metadata['group_details'].append(group_meta)
    
    # Calculate overall statistics
    nonzero_probs = probability_array[probability_array > 0]
    if len(nonzero_probs) > 0:
        metadata['overall_proportion'] = np.mean(nonzero_probs)
        metadata['max_proportion'] = np.max(nonzero_probs)
        metadata['min_proportion'] = np.min(nonzero_probs)
    else:
        metadata['overall_proportion'] = 0
        metadata['max_proportion'] = 0
        metadata['min_proportion'] = 0
    
    return metadata

# Primary function that runs all of the rest
def generate_landslide_probability(grid, h_pga_array, v_pga_array, labeled_array, 
                                   slope_array=None, soil_array=None, 
                                   geological_factor_array=None, critical_acceleration_array=None,
                                   default_critical_acceleration=0.2, random_seed=None,
                                   normalise_final_probs=False):
    """
    Generate landslide probability estimation with critical acceleration consideration.
    
    Parameters:
    -----------
    grid : ndarray or tuple
        Grid shape reference for reshaping arrays
    h_pga_array : ndarray
        Horizontal Peak Ground Acceleration array
    v_pga_array : ndarray
        Vertical Peak Ground Acceleration array
    labeled_array : ndarray
        Labeled regions for analysis
    slope_array : ndarray, optional
        Slope angles in degrees
    soil_array : ndarray, optional
        Soil susceptibility index
    geological_factor_array : ndarray, optional
        Geological instability factors
    critical_acceleration_array : ndarray, optional
        Critical acceleration thresholds for each region
    default_critical_acceleration : float, optional
        Fallback critical acceleration value
    random_seed : int, optional
        Seed for reproducibility
    normalise_final_probs : bool, optional
        Select whether final probabilities will be normalised
    
    Returns:
    --------
    probability_array : ndarray
        Landslide failure probabilities
    metadata : dict
        Detailed analysis metadata
    """
    # Setup and reshape input arrays
    h_pga_grid = h_pga_array.reshape(grid.shape)
    v_pga_grid = v_pga_array.reshape(grid.shape)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Prepare critical acceleration array
    if critical_acceleration_array is None:
        critical_acceleration_grid = np.full_like(
            h_pga_grid, 
            default_critical_acceleration, 
            dtype=np.float32
        )
    else:
        critical_acceleration_grid = critical_acceleration_array.reshape(grid.shape)
    
    # Prepare slope grid if available
    slope_grid = None
    if slope_array is not None:
        slope_grid = slope_array.reshape(grid.shape)
    
    # Initialize output array
    unique_labels = np.unique(labeled_array)[1:]  # Exclude zero
    probability_array = np.zeros_like(labeled_array, dtype=np.float32)
    
    # Calculate probabilities for each group
    group_probs = {}
    for label_num in unique_labels:
        mask = labeled_array == label_num
        
        # Calculate probability for this group
        group_info = calculate_group_probability(
            h_pga_grid, v_pga_grid, critical_acceleration_grid,
            mask, slope_grid, soil_array, geological_factor_array
        )
        
        # Store mask for later use
        group_info['mask'] = mask
        group_probs[label_num] = group_info
    
    # Normalize if requested
    if normalise_final_probs:
        normalized_group_probs, norm_metadata = normalize_group_probabilities(group_probs)
        
        # Apply normalized probabilities to output array
        probability_array = apply_probabilities_to_array(
            probability_array, normalized_group_probs, normalized=True
        )
        
        # Create metadata
        metadata = create_metadata(
            normalized_group_probs, probability_array, norm_metadata, normalized=True
        )
    else:
        # Apply raw probabilities to output array
        probability_array = apply_probabilities_to_array(
            probability_array, group_probs, normalized=False
        )
        
        # Create metadata without normalization
        metadata = create_metadata(
            group_probs, probability_array, {'performed': False}, normalized=False
        )
    
    return probability_array, metadata

# %%%%% Proportion calculation and group selection
def calculate_landslide_proportion(probability_array, method='empirical'):
    """
    Dynamically calculate an appropriate proportion of landslide groups.
    
    Parameters:
    -----------
    probability_array : numpy.ndarray
        Array of failure probabilities
    method : str, optional
        Method for proportion calculation
    
    Returns:
    --------
    float
        Recommended proportion of landslide groups
    """
    # Remove zero probabilities
    valid_probs = np.unique(probability_array[probability_array > 0])
    
    if method == 'empirical':
        # Method 1: Based on probability distribution percentiles
        # Uses 75th percentile as a natural breaking point
        proportion = np.percentile(valid_probs, 75)
        normalized_proportion = proportion / np.max(valid_probs)
        
        return normalized_proportion
    
    elif method == 'statistical':
        # Method 2: Using statistical distribution characteristics
        mean_prob = np.mean(valid_probs)
        std_prob = np.std(valid_probs)
        
        # One standard deviation above the mean as a threshold
        threshold = mean_prob + std_prob
        proportion = np.sum(probability_array >= threshold) / probability_array.size
        
        return proportion
    
    elif method == 'risk_profile':
        # Method 3: Multi-factor risk assessment
        # Considers both probability and variability
        mean_prob = np.mean(valid_probs)
        median_prob = np.median(valid_probs)
        std_prob = np.std(valid_probs)
        
        # Combine metrics with weights and ensure result is between 0 and 1
        risk_score = (
            0.4 * (mean_prob / np.max(valid_probs)) + 
            0.3 * (median_prob / np.max(valid_probs)) + 
            0.3 * np.clip(std_prob / mean_prob, 0, 1)
        )
        
        # Ensure the final proportion is between 0 and 1
        return np.clip(risk_score, 0, 1)
    
    else:
        raise ValueError("Invalid method. Choose 'empirical', 'statistical', or 'risk_profile'.")

def probabilistic_group_selection(labeled_array, probability_array, proportion_method='empirical', 
                                  custom_proportion=None, random_seed=None):
    """
    Enhanced probabilistic group selection with dynamic proportion calculation.
    
    Parameters:
    -----------
    probability_array : ndarray
        Array of failure probabilities
    proportion_method : str, optional
        Method to calculate proportion
    custom_proportion : float, optional
        Override calculated proportion with a specific value
    random_seed : int, optional
        Seed for reproducibility
    
    Returns:
    --------
    selected_groups : ndarray
        Selected groups matching input array shape
    metadata : dict
        Metadata about proportion selection
    """
    
    unique_labels = np.unique(labeled_array)
    unique_labels = unique_labels[unique_labels != 0]
    num_groups = len(unique_labels)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Determine proportion
    if custom_proportion is not None:
        proportion = custom_proportion
        method = 'user_defined'
    else:
        proportion = calculate_landslide_proportion(
            probability_array, 
            method=proportion_method
        )
    
    # Get probabilities for each group
    group_probs = []
    for label_name in unique_labels:
        mask = (labeled_array == label_name)
        group_prob = np.mean(probability_array[mask])
        group_probs.append(group_prob)
    
    group_probs = np.array(group_probs)
    # Normalize probabilities
    group_probs = group_probs / np.sum(group_probs)
    
    num_to_select = int(np.ceil(num_groups*proportion))
    
    # Select groups based on probabilities
    selected_labels = np.random.choice(
        unique_labels, num_to_select, replace=False, p=group_probs)
    
    selected_groups = np.isin(labeled_array, selected_labels) * labeled_array
    
    return selected_groups, proportion

# %%% Method 3: Select groups/proportion from PGA
def generate_landslide_proportion_from_pga(grid, h_pga, v_pga, labeled_array, 
                                         weight_array=None, slope_array=None, 
                                         soil_condition_array=None, random_seed=None):
    """
    Generates landslide probability using both horizontal and vertical PGA components.
    
    Parameters
    ----------
    h_pga_array : array
        Array containing horizontal PGA values in g.
    v_pga_array : array
        Array containing vertical PGA values in g.
    labeled_array : array
        Array of labeled regions (unstable areas).
    weight_array : array, optional
        Array containing weights for group selection. Values closer to zero 
        correspond to higher selection probabilities.
    slope_array : array, optional
        Array containing slope angles in degrees.
    soil_condition_array : array, optional
        Array containing soil susceptibility factors (0-1).
    random_seed : int, optional
        Seed for random number generator.
    
    Returns
    -------
    probability_array : array
        Array containing probability of failure for each labeled group.
    proportion : float
        Overall proportion of unstable areas predicted to fail.
    metadata : dict
        Dictionary containing additional information about the calculations.
    """
    
    h_pga_array = h_pga.reshape(grid.shape)
    v_pga_array = v_pga.reshape(grid.shape)
    
    # Uses a value for the random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"Random seed = {random_seed}")
    
    unique_labels = np.unique(labeled_array)
    unique_labels = unique_labels[unique_labels != 0]
    
    group_probabilities = []
    probability_array = np.zeros_like(labeled_array, dtype=np.float32)
    
    # Store metadata for analysis
    metadata = {
        'group_data': [],
        'mean_h_pga': np.mean(h_pga_array),
        'mean_v_pga': np.mean(v_pga_array),
        'num_groups': len(unique_labels)
    }
    
    for label_name in unique_labels:
        mask = (labeled_array == label_name)
        
        # Get PGA values for this group
        group_h_pga = h_pga_array[mask]
        group_v_pga = v_pga_array[mask]
        
        # Calculate mean PGA values
        mean_h_pga = np.mean(group_h_pga)
        mean_v_pga = np.mean(group_v_pga)
        
        # Calculate vector resultant PGA
        resultant_pga = np.sqrt(mean_h_pga**2 + mean_v_pga**2)
        
        # Calculate V/H ratio
        vh_ratio = mean_v_pga / mean_h_pga if mean_h_pga > 0 else 0
        
        # Base probability calculation considering both components
        h_prob = calculate_prob_from_h_pga(mean_h_pga)
        r_prob = calculate_prob_from_resultant(resultant_pga, vh_ratio)
        base_prob = 0.7 * h_prob + 0.3 * r_prob
        
        # Apply slope factor if available
        slope_factor = 1.0
        if slope_array is not None:
            group_slope = slope_array[mask]
            mean_slope = np.mean(group_slope)
            slope_factor = calculate_slope_factor(mean_slope)
            base_prob *= slope_factor
        
        # Apply soil conditions if available
        soil_factor = 1.0
        if soil_condition_array is not None:
            group_soil = soil_condition_array[mask]
            mean_soil = np.mean(group_soil)
            soil_factor = 0.5 + 0.5 * mean_soil
            base_prob *= soil_factor
            
        # Apply weight factor (from critical acceleration)
        if weight_array is not None:
            group_weight = np.mean(weight_array[mask])
            epsilon = 1e-10
            weight_factor = 1.0 / (group_weight + epsilon)
            base_prob *= weight_factor
        
        # Add stochastic component
        stochastic_factor = np.random.lognormal(mean=0, sigma=0.3)
        group_prob = base_prob * stochastic_factor
        
        # Ensure probability is between 0 and 1
        group_prob = np.clip(group_prob, 0.0, 1.0)
        
        # Store group probability
        group_probabilities.append(group_prob)
        probability_array[mask] = group_prob
        
        # Store metadata for this group
        metadata['group_data'].append({
            'label': label_name,
            'mean_h_pga': mean_h_pga,
            'mean_v_pga': mean_v_pga,
            'resultant_pga': resultant_pga,
            'vh_ratio': vh_ratio,
            'h_prob': h_prob,
            'r_prob': r_prob,
            'base_prob': base_prob,
            'slope_factor': slope_factor,
            'soil_factor': soil_factor,
            'weight_factor': weight_factor,
            'final_prob': group_prob
        })
    
    proportion = np.mean(group_probabilities)
    metadata['overall_proportion'] = proportion
    
    return probability_array, proportion, metadata

def select_groups_by_proportion_weighted(labeled_array, probability_array, proportion=None):
    """
    Selects a specified proportion of groups from the labeled array using probabilities.
    
    Parameters
    ----------
    labeled_array : array of int32
        Array of labeled regions.
    probability_array : array of same shape as labeled_array
        Array containing probability values for each group.
    proportion : float, optional
        Proportion of groups to select (between 0 and 1). If None, uses probabilities
        directly.
    
    Returns
    -------
    selected_groups : array of int32
        Array with only the selected groups.
    selected_labels : list of int
        List of selected group labels.
    """
    unique_labels = np.unique(labeled_array)
    unique_labels = unique_labels[unique_labels != 0]
    num_groups = len(unique_labels)
    
    if proportion is not None:
        num_to_select = int(np.ceil(proportion * num_groups))
    else:
        num_to_select = num_groups  # Will use probabilities to determine selection
    
    # Get probabilities for each group
    group_probs = []
    for label_name in unique_labels:
        mask = (labeled_array == label_name)
        group_prob = np.mean(probability_array[mask])
        group_probs.append(group_prob)
    
    group_probs = np.array(group_probs)
    
    # Normalize probabilities
    group_probs = group_probs / np.sum(group_probs)
    
    # Select groups based on probability array
    selected_labels = np.random.choice(
        unique_labels, num_to_select, replace=False, p=group_probs)
    
    selected_groups = np.isin(labeled_array, selected_labels) * labeled_array
    
    return selected_groups, selected_labels

def calculate_prob_from_h_pga(h_pga):
    """
    Calculate probability based on horizontal PGA.
    Based on Jibson (2007) relationships.
    
    Parameters
    ----------
    h_pga : float
        Horizontal PGA value in g.
    
    Returns
    -------
    float
        Probability value.
    """
    if h_pga < 0.05:
        return 0.01 * (h_pga / 0.05)
    else:
        return 0.01 + 0.3 * (h_pga - 0.05)

def calculate_prob_from_resultant(resultant_pga, vh_ratio):
    """
    Calculate probability based on resultant PGA and V/H ratio.
    
    Parameters
    ----------
    resultant_pga : float
        Vector resultant of horizontal and vertical PGA.
    vh_ratio : float
        Ratio of vertical to horizontal PGA.
    
    Returns
    -------
    float
        Probability value.
    """
    base_prob = calculate_prob_from_h_pga(resultant_pga)
    
    # Modify based on V/H ratio - higher ratios can increase probability
    if vh_ratio > 0.5:  # Significant vertical component
        vh_factor = 1.0 + 0.2 * (vh_ratio - 0.5)
        base_prob *= min(vh_factor, 1.5)  # Cap the increase at 50%
    
    return base_prob

def calculate_slope_factor(slope):
    """
    Calculate slope factor based on slope angle.
    
    Parameters
    ----------
    slope : float
        Slope angle in degrees.
    
    Returns
    -------
    float
        Slope factor value.
    """
    if slope < 15:
        return 0.1 + 0.03 * slope
    else:
        return 0.1 + 0.03 * 15 + 0.08 * (slope - 15)
# %%% Calculate properties for selected regions
def calculate_region_properties(grid, labeled_array, slopes, aspect_array, min_size=1, handle_small='keep'):
    """
    Calculates various geometric properties for each of the identified regions

    Parameters
    ----------
    grid : landlab.grid
        Landlab grid containing elevation data
    labeled_array : ndarray
        Numpy array with each of the labeled regions 
    slopes : _type_
        Numpy array of slope values
    aspect_array : _type_
        Numpy array of topographic aspect values
    min_size : int, optional
        Minimum integer number of pixels that a region can have, by default 1
    handle_small : str, optional
        Flag to choose how to manage small (< min_size) regions, by default 'keep'
        Can be: 'keep', 'merge', 'remove'

    Returns
    -------
    propos : Pandas dataframe
        Dataframe listing geometric properties for each labeled region
        Properties include:
            region label
            area
            max, min elevation
            median elevation
            local relief
            mean topographic aspect
            perimeter
            compactness
            bounding box (bbox) width, height, area
            fill ratio
            major, minor axis lengths
            orientation of major axis
            eccentricity
            slope-parallel length
            slope-perpendicular width
            hybrid length, width

    Raises
    ------
    ValueError
        Raised if array of labeled regions is not the same size as the landlab grid
    """


    if labeled_array.shape != (grid.number_of_node_rows, grid.number_of_node_columns):
        raise ValueError("Labeled array must match grid dimensions")

    if handle_small in ['merge', 'remove'] and min_size > 1:
        working_labeled_array = labeled_array.copy()
        working_labeled_array = handle_small_regions(working_labeled_array, min_size, handle_small, grid)
    else:
        working_labeled_array = labeled_array

    unique_labels = np.unique(working_labeled_array)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return pd.DataFrame(), working_labeled_array

    elevation_grid = grid.at_node['topographic__elevation'].reshape(grid.shape)
    slopes_grid = slopes.reshape(grid.shape)

    props = {
        'label': unique_labels,
        'area': np.zeros_like(unique_labels, dtype=float),
        'max_elevation': np.zeros_like(unique_labels, dtype=float),
        'median_elevation': np.zeros_like(unique_labels, dtype=float),
        'local_relief': np.zeros_like(unique_labels, dtype=float),
        'median_slope': np.zeros_like(unique_labels, dtype=float),
        'mean_aspect': np.zeros_like(unique_labels, dtype=float),
        'perimeter': np.zeros_like(unique_labels, dtype=float),
        'compactness': np.zeros_like(unique_labels, dtype=float),
        'bbox_width': np.zeros_like(unique_labels, dtype=float),
        'bbox_height': np.zeros_like(unique_labels, dtype=float),
        'bbox_area': np.zeros_like(unique_labels, dtype=float),
        'fill_ratio': np.zeros_like(unique_labels, dtype=float),
        'major_axis_length': np.zeros_like(unique_labels, dtype=float),
        'minor_axis_length': np.zeros_like(unique_labels, dtype=float),
        'orientation': np.zeros_like(unique_labels, dtype=float),
        'eccentricity': np.zeros_like(unique_labels, dtype=float),
        'slope_direction_length': np.zeros_like(unique_labels, dtype=float),
        'perpendicular_width': np.zeros_like(unique_labels, dtype=float),
        'hybrid_length': np.zeros_like(unique_labels, dtype=float),
        'hybrid_width': np.zeros_like(unique_labels, dtype=float)
    }

    calculate_topographic_stats(props, unique_labels, working_labeled_array, elevation_grid, slopes_grid, aspect_array)
    region_properties = regionprops(working_labeled_array)
    calculate_geometric_properties(props, unique_labels, working_labeled_array, region_properties, grid)
    calculate_slope_direction_metrics(props, unique_labels, working_labeled_array, grid)
    calculate_landslide_shape_metrics(props)

    props_df = pd.DataFrame(props)
    props_df.set_index('label', inplace=True)

    return props_df, working_labeled_array

# %%%% Helper functions
def handle_small_regions(labeled_array, min_size, method='merge', grid=None):
    """
    Helper function to handle labeled regions that are smaller than a defined threshold minimum size
    
    Parameters
    ----------
    
    labeled_array : ndarray
        Numpy array with each of the labeled regions
    min_size : int, optional
        Minimum integer number of pixels that a region can have, by default 1
    handle_small : str, optional
        Flag to choose how to manage small (< min_size) regions, by default 'merge'
        Can be: 'keep', 'merge', 'remove'
    grid : landlab.grid
        Landlab grid containing elevation data
        
    
    """

    if method not in ['merge', 'remove']:
        return labeled_array

    print('Processing small regions...')
    modified_array = labeled_array.copy()
    props = regionprops(labeled_array)
    small_regions = [
        {'label': region.label, 'mask': labeled_array == region.label, 'centroid': region.centroid, 'size': region.area}
        for region in props if region.area < min_size
    ]

    if not small_regions:
        return labeled_array

    if method == 'remove':
        for region in small_regions:
            modified_array[region['mask']] = 0
        return modified_array

    small_regions.sort(key=lambda x: x['size'])
    for region in small_regions:
        mask = region['mask']
        dilated = binary_dilation(mask)
        neighbor_mask = dilated & ~mask
        neighbor_labels = np.unique(modified_array[neighbor_mask])
        neighbor_labels = neighbor_labels[(neighbor_labels > 0) & (neighbor_labels != region['label'])]

        if len(neighbor_labels) > 0:
            neighbor_counts = [(nl, np.sum(modified_array[neighbor_mask] == nl)) for nl in neighbor_labels]
            best_neighbor = max(neighbor_counts, key=lambda x: x[1])[0]
            modified_array[mask] = best_neighbor

    return modified_array


def calculate_topographic_stats(props, unique_labels, labeled_array, elevation_grid, slopes_grid, aspect_array):
    props['median_elevation'] = labeled_comprehension(elevation_grid, labeled_array, unique_labels, np.median, float, 0)
    props['max_elevation'] = labeled_comprehension(elevation_grid, labeled_array, unique_labels, np.max, float, 0)
    props['local_relief'] = props['max_elevation'] - labeled_comprehension(elevation_grid, labeled_array, unique_labels, np.min, float, 0)
    props['median_slope'] = labeled_comprehension(slopes_grid, labeled_array, unique_labels, np.median, float, 0)
    props['mean_aspect'] = labeled_comprehension(aspect_array, labeled_array, unique_labels, np.mean, float, 0)


def calculate_geometric_properties(props, unique_labels, labeled_array, region_properties, grid):
    for i, label_num in enumerate(unique_labels):
        region_mask = labeled_array == label_num
        props['area'][i] = np.sum(region_mask) * grid.dx * grid.dy
        props['perimeter'][i] = calculate_perimeter_vectorized(region_mask, grid.dx, grid.dy)
        if props['perimeter'][i] > 0:
            props['compactness'][i] = 4 * np.pi * props['area'][i] / (props['perimeter'][i] ** 2)

        region_idx = next((j for j, r in enumerate(region_properties) if r.label == label_num), None)
        if region_idx is not None:
            extract_region_shape_metrics(props, i, region_properties[region_idx], grid)


def extract_region_shape_metrics(props, i, region, grid):
    min_row, min_col, max_row, max_col = region.bbox
    props['bbox_height'][i] = (max_row - min_row) * grid.dy
    props['bbox_width'][i] = (max_col - min_col) * grid.dx
    props['bbox_area'][i] = props['bbox_height'][i] * props['bbox_width'][i]
    if props['bbox_area'][i] > 0:
        props['fill_ratio'][i] = props['area'][i] / props['bbox_area'][i]

    epsilon = 1
    props['major_axis_length'][i] = region.major_axis_length * grid.dx
    props['minor_axis_length'][i] = region.minor_axis_length * grid.dx
    if props['minor_axis_length'][i] == 0:
        props['minor_axis_length'][i] += epsilon

    props['orientation'][i] = region.orientation * (180 / np.pi)
    props['eccentricity'][i] = region.eccentricity


def calculate_slope_direction_metrics(props, unique_labels, labeled_array, grid):
    for i, label_num in enumerate(unique_labels):
        region_mask = labeled_array == label_num
        mean_aspect_radians = props['mean_aspect'][i] * (np.pi / 180)
        slope_direction = np.array([np.cos(mean_aspect_radians), np.sin(mean_aspect_radians)])
        slope_direction /= np.linalg.norm(slope_direction)
        perp_direction = np.array([-slope_direction[1], slope_direction[0]])

        coords = np.column_stack(np.where(region_mask))
        if len(coords) > 0:
            map_coords = coords * np.array([grid.dy, grid.dx])
            centroid = np.mean(map_coords, axis=0)
            centered_coords = map_coords - centroid

            slope_proj = np.dot(centered_coords, slope_direction)
            perp_proj = np.dot(centered_coords, perp_direction)

            props['slope_direction_length'][i] = np.max(slope_proj) - np.min(slope_proj)
            props['perpendicular_width'][i] = np.max(perp_proj) - np.min(perp_proj)


def calculate_landslide_shape_metrics(props):
    for i in range(len(props['label'])):
        length_candidates = [
            props['slope_direction_length'][i],
            props['major_axis_length'][i],
            max(props['bbox_height'][i], props['bbox_width'][i])
        ]
        width_candidates = [
            props['perpendicular_width'][i],
            props['minor_axis_length'][i],
            min(props['bbox_height'][i], props['bbox_width'][i])
        ]

        props['hybrid_length'][i] = np.median(length_candidates)
        props['hybrid_width'][i] = np.median(width_candidates)


def calculate_perimeter_vectorized(region_mask, dx, dy):
    dilated = binary_dilation(region_mask)
    boundary = np.logical_and(dilated, ~region_mask)
    perimeter_length = np.sum(boundary) * (dx + dy) / 2
    return perimeter_length
# %%% Split regions by width
def split_wide_regions(labeled_array, region_df, kde_results, transform_info,
                       width_threshold=1.5, label_col='label', 
                       length_col='length_m', width_col='width_m'):
    """
    Split labeled regions where the actual width is significantly larger than
    the width expected from the KDE distribution, using a simpler approach
    that focuses only on regions that need splitting.
    
    Parameters:
    -----------
    labeled_array : numpy.ndarray
        2D array of integer labels where each region has a unique label
    region_df : pandas.DataFrame
        DataFrame containing measurements for each region
    kde_results : dict
        Dictionary with KDE objects from fit_bivariate_kde function
    transform_info : dict
        Information about transformations used in the KDE
    width_threshold : float, optional
        Ratio of actual width to KDE-expected width above which regions are split (default: 1.5)
    label_col : str, optional
        Name of the column in region_df containing region labels (default: 'label')
    length_col : str, optional
        Name of the column in region_df containing region lengths (default: 'length_m')
    width_col : str, optional
        Name of the column in region_df containing region widths (default: 'width_m')
        
    Returns:
    --------
    numpy.ndarray
        New array where wide regions have been split
    list
        Information about the split regions
    """
    import numpy as np
    from tqdm import tqdm
    
    region_df = region_df.reset_index()
    
    # Verify the column names exist in the DataFrame
    for col_name, col_desc in [(label_col, "label"), (length_col, "length"), (width_col, "width")]:
        if col_name not in region_df.columns:
            raise ValueError(f"Column '{col_name}' specified for {col_desc} not found in the DataFrame. "
                           f"Available columns are: {list(region_df.columns)}")
    
    # Start with a copy of the original labeled array
    new_labels = labeled_array.copy()
    next_label = np.max(labeled_array) + 1 if labeled_array.size > 0 else 1
    split_info = []
    
    # Get KDE information
    kde = kde_results['overall']
    log_x = transform_info.get('log_x', False)
    log_y = transform_info.get('log_y', False)
    
    # First, identify which regions need splitting
    regions_to_split = []
    
    print("Identifying regions to split...")
    for _, row in tqdm(region_df.iterrows(), total=len(region_df)):
        label_id = row[label_col]
        if label_id == 0:  # Skip background
            continue
            
        length = row[length_col]
        actual_width = row[width_col]
        
        # Skip if length or width are invalid
        if length <= 0 or actual_width <= 0:
            continue
        
        # Transform length value if needed (to match KDE space)
        if log_x:
            length_t = np.log(length)
        else:
            length_t = length
            
        # Sample from KDE to get expected width for this length
        num_samples = 200  # Reduced number of samples for efficiency
        samples = []
        attempts = 0
        max_attempts = 500
        
        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1
            kde_samples = kde.resample(1).T
            # Keep only samples with length close to our target
            if abs(kde_samples[0, 0] - length_t) < 0.05 * (1 + abs(length_t)):
                samples.append(kde_samples[0, 1])
        
        # If we couldn't get enough samples, use what we have
        if len(samples) < 10:
            # Just generate some directly - not conditioned but better than failing
            samples = kde.resample(num_samples)[1, :]
            
        # Convert back from transformed space if needed
        if log_y:
            expected_widths = np.exp(samples)
        else:
            expected_widths = samples
            
        expected_width = np.median(expected_widths)
        
        # Check if this region needs splitting
        width_ratio = actual_width / expected_width
        if width_ratio > width_threshold:
            regions_to_split.append({
                'label': label_id,
                'actual_width': actual_width,
                'expected_width': expected_width,
                'ratio': width_ratio
            })
    
    # Then split only those regions
    print(f"Splitting {len(regions_to_split)} regions...")
    for region in tqdm(regions_to_split):
        label_id = region['label']
        
        # Get region mask
        mask = labeled_array == label_id
        
        # Find coordinates and centroid
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            continue  # Skip empty regions
            
        centroid_x = np.mean(x_coords)
        
        # Create a split threshold based on the centroid x-coordinate
        # Instead of trying to index with x_coords directly, we'll create
        # new masks by checking each pixel's position against the threshold
        
        # CORRECTED APPROACH: Create the split using a direct comparison
        # with the mask's position relative to the centroid
        rows, cols = np.indices(labeled_array.shape)
        
        # Create left and right masks - use the full indices array
        left_mask = mask & (cols < centroid_x)
        right_mask = mask & (cols >= centroid_x)
        
        # Ensure both parts have pixels
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            continue  # Skip if one half would be empty
        
        # Replace with new labels
        new_labels[left_mask] = next_label
        left_label = next_label
        next_label += 1
        
        new_labels[right_mask] = next_label
        right_label = next_label
        next_label += 1
        
        # Record the split
        region['left_label'] = left_label
        region['right_label'] = right_label
        region['original_size'] = len(x_coords)
        region['left_size'] = np.sum(left_mask)
        region['right_size'] = np.sum(right_mask)
        split_info.append(region)
    
    print(f"Split {len(split_info)} regions successfully.")
    return new_labels, split_info


def sample_kde_widths(kde_results, transform_info, length_array, n_samples=100):
    """
    For an array of lengths, sample expected widths from a KDE.
    Useful for getting expected widths for many regions at once.
    
    Parameters:
    -----------
    kde_results : dict
        KDE results from fit_bivariate_kde
    transform_info : dict
        Transform information from fit_bivariate_kde
    length_array : array-like
        Array of length values to sample widths for
    n_samples : int, optional
        Number of samples to generate for each length (default: 100)
        
    Returns:
    --------
    numpy.ndarray
        Array of expected widths corresponding to each input length
    """
    import numpy as np
    
    kde = kde_results['overall']
    log_x = transform_info.get('log_x', False)
    log_y = transform_info.get('log_y', False)
    
    expected_widths = np.zeros_like(length_array, dtype=float)
    
    for i, length in enumerate(length_array):
        # Transform length value if needed
        if log_x:
            length_t = np.log(length) if length > 0 else 0
        else:
            length_t = length
            
        # Sample from KDE
        samples = []
        attempts = 0
        max_attempts = 500
        
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            kde_samples = kde.resample(1).T
            # Keep only samples with length close to our target
            if abs(kde_samples[0, 0] - length_t) < 0.05 * (1 + abs(length_t)):
                samples.append(kde_samples[0, 1])
        
        # If we couldn't get enough samples, use what we have
        if len(samples) < 10:
            # Just generate some directly - not conditioned but better than failing
            samples = kde.resample(n_samples)[1, :]
            
        # Convert back from transformed space if needed
        if log_y:
            samples_widths = np.exp(samples)
        else:
            samples_widths = samples
            
        expected_widths[i] = np.median(samples_widths)
    
    return expected_widths
# %%%% Calculate length and width according to slope
# TODO: Is this needed?
def calculate_region_dimensions(grid, labels, slopes, aspects):
    
    slopes_grid = slopes.reshape(grid.shape)
    
    # Get unique region labels (excluding background if labeled as 0)
    unique_labels = np.unique(labels)
    if 0 in unique_labels:
        unique_labels = unique_labels[1:]
    
    results = {
        'label': unique_labels,
        'length': np.zeros_like(unique_labels, dtype=float),
        'width': np.zeros_like(unique_labels, dtype=float),
        'aspect_degrees': np.zeros_like(unique_labels, dtype=float)
        }
    
    for label_num in unique_labels:
        # Get mask and coordinates for this region
        mask = labels == label_num
        coords = np.column_stack(np.where(mask))
        
        # Calculate average slope direction (convert aspect to radians)
        region_aspects = aspects[mask]
        region_slopes = slopes_grid[mask]
        
        # Weighted by slope magnitude to give more importance to steeper areas
        avg_aspect_rad = np.average(np.radians(region_aspects), weights=region_slopes)
        
        # Convert to unit vector pointing downslope
        downslope_vector = np.array([np.sin(avg_aspect_rad), np.cos(avg_aspect_rad)])
        
        # Define perpendicular vector (orthogonal to downslope)
        perp_vector = np.array([-downslope_vector[1], downslope_vector[0]])
        
        # Project coordinates onto slope direction and perpendicular direction
        slope_proj = np.dot(coords, downslope_vector)
        perp_proj = np.dot(coords, perp_vector)
        
        # Calculate dimensions
        length = np.max(slope_proj) - np.min(slope_proj)
        width = np.max(perp_proj) - np.min(perp_proj)
        
        # Convert to actual grid units if needed
        length *= grid.dx  # Assuming grid.dx is the node spacing
        width *= grid.dx   # Adjust as needed for your grid
        
        results['length'] = length
        results['width'] = width
        results['aspect_degrees'] = np.degrees(avg_aspect_rad)
    
    results_df = pd.DataFrame(results)
    results_df.set_index(keys='label')
    
    return results_df

# %% ### Calculation of instability ###
# %%% Generate soil depth array
def apply_soil_depth(grid,
                    elevation_field='topographic__elevation', 
                    soil_field='soil__depth',
                    max_soil_depth=1.0,
                    distribution='uniform',
                    plot=False
                    ):
    """
    Apply soil depth to core nodes based on elevation or uniform distribution.
    
    Parameters:
    -----------
    grid : Landlab grid object
        The landlab grid (RasterModelGrid, HexModelGrid, etc.)
    elevation_field : str, default 'topographic__elevation'
        Name of the field containing elevation data
    soil_field : str, default 'soil__depth'
        Name of the field to store soil depth data
    max_soil_depth : float, default 1.0
        Maximum soil depth in meters. For elevation-based: applied to minimum elevation.
        For uniform: applied to all core nodes.
    distribution : str, default 'uniform'
        When 'uniform' : 
        - All core nodes get the same soil depth (max_soil_depth)
        - Boundary nodes get zero soil depthapplies the max_soil_depth uniformly across entire grid
        When 'elevation' :
        - Soil depth varies inversely with elevation
        - Minimum elevation gets maximum soil depth
        - Maximum elevation gets zero soil depth
    plot : bool, default False
        If True, creates a scatter plot showing soil depth vs elevation relationship.
    
    Returns:
    --------
    soil_depth : ndarray
        1D array of soil depth values, associated with the grid 'soil__depth' field
        
    """
    # Get elevation data and core nodes
    elevation = grid.at_node[elevation_field]
    core_nodes = grid.core_nodes
    
    # Initialize soil depth field on the grid (zeros everywhere)
    soil_depth = grid.add_zeros("node", soil_field, clobber=True)
    
    if distribution == 'uniform':
        # Apply uniform soil depth to core nodes only
        soil_depth[core_nodes] = max_soil_depth
        print(f"Uniform soil depth applied to grid:")
        print(f"  Soil depth: {max_soil_depth:.2f} m (uniform)")
        print(f"  Core nodes processed: {len(core_nodes)}")
        
    elif distribution == 'elevation':
        # Apply elevation-based soil depth
        core_elevations = elevation[core_nodes]
        
        # Calculate min and max elevation from core nodes
        min_elevation = np.min(core_elevations)
        max_elevation = np.max(core_elevations)
        
        # Check if there's any elevation variation
        elevation_range = max_elevation - min_elevation
        if elevation_range == 0:
            # If all elevations are the same, assign uniform soil depth
            soil_depth[core_nodes] = max_soil_depth / 2.0
            print("Warning: All core nodes have the same elevation. Assigning uniform soil depth.")
        else:
            # Calculate normalized elevation (0 = min elevation, 1 = max elevation)
            normalized_elevation = (elevation - min_elevation) / elevation_range
            
            # Apply inverse relationship to all nodes first
            temp_soil_depth = max_soil_depth * (1.0 - normalized_elevation)
            
            # Ensure non-negative values
            temp_soil_depth = np.maximum(temp_soil_depth, 0.0)
            
            # Copy values to the grid field
            soil_depth[:] = temp_soil_depth
            
            # Set boundary nodes to zero
            boundary_nodes = np.setdiff1d(np.arange(grid.number_of_nodes), core_nodes)
            soil_depth[boundary_nodes] = 0.0
        
        # Print summary statistics
        print(f"Elevation-based soil depth applied to grid:")
        print(f"  Elevation range: {min_elevation:.2f} to {max_elevation:.2f} m")
        print(f"  Soil depth range: 0.00 to {max_soil_depth:.2f} m")
        print(f"  Core nodes processed: {len(core_nodes)}")
    
    else:
        raise ValueError("Soil distribution can only be 'uniform' or 'elevation'")
    
    # Create plot if requested
    if plot:
        create_soil_depth_plot(grid, elevation_field, soil_field, uniform)
    
    return soil_depth

# Helper function to create plot of soil depth distribution
def create_soil_depth_plot(grid,
                           elevation_field,
                           soil_field,
                           uniform
                           ):
    """
    Create a scatter plot showing the relationship between elevation and soil depth.
    
    Parameters:
    -----------
    grid : Landlab grid object
        The landlab grid with elevation and soil depth fields
    elevation_field : str
        Name of the elevation field
    soil_field : str
        Name of the soil depth field
    uniform : bool
        Whether uniform soil depth was applied
    """
    # Get data for core nodes only
    elevation = grid.at_node[elevation_field]
    soil_depth = grid.at_node[soil_field]
    core_nodes = grid.core_nodes
    
    core_elevation = elevation[core_nodes]
    core_soil_depth = soil_depth[core_nodes]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    if uniform:
        # For uniform case, show horizontal line
        plt.scatter(core_elevation, core_soil_depth, alpha=0.6, s=30, color='blue', 
                   label=f'Core nodes (n={len(core_nodes)})')
        plt.axhline(y=core_soil_depth[0], color='red', linestyle='--', alpha=0.7,
                   label=f'Uniform depth = {core_soil_depth[0]:.2f} m')
        plt.title('Uniform Soil Depth Distribution')
    else:
        # For elevation-based case, show scatter with trend line
        plt.scatter(core_elevation, core_soil_depth, alpha=0.6, s=30, color='blue',
                   label=f'Core nodes (n={len(core_nodes)})')
        
        # Add theoretical trend line
        elev_range = np.linspace(core_elevation.min(), core_elevation.max(), 100)
        max_depth = core_soil_depth.max()
        min_elev = core_elevation.min()
        max_elev = core_elevation.max()
        
        if max_elev > min_elev:  # Avoid division by zero
            normalized_elev = (elev_range - min_elev) / (max_elev - min_elev)
            theoretical_depth = max_depth * (1.0 - normalized_elev)
            plt.plot(elev_range, theoretical_depth, 'r--', alpha=0.7, 
                    label='Theoretical relationship')
        
        plt.title('Elevation-Based Soil Depth Distribution')
    
    plt.xlabel('Elevation (m)')
    plt.ylabel('Soil Depth (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Elevation: {core_elevation.min():.2f} - {core_elevation.max():.2f} m\n'
    stats_text += f'Soil depth: {core_soil_depth.min():.2f} - {core_soil_depth.max():.2f} m\n'
    stats_text += f'Mean soil depth: {core_soil_depth.mean():.2f} m'
    
    plt.text(0.02, 0.22, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
# %%% Generate earthquakes
def generate_acceleration_grid(grid, max_horizontal, max_vertical,
                               distribution="uniform", center=None,
                               random_center=False, seed=None):
    """
    Generate arrays of horizontal and vertical acceleration values for a landlab grid.
    
    Parameters:
    ----------
    grid : RasterModelGrid
        The landlab grid to generate acceleration values for
    max_horizontal : float
        Maximum horizontal acceleration value at the center
    max_vertical : float
        Maximum vertical acceleration value at the center
    distribution : str, optional
        Distribution shape: "uniform", "circular", "square", "diamond", or "exponential" (default: "uniform")
    center : tuple, optional
        (row, col) coordinates of the center point. If None and random_center is False, 
        the center of the grid is used.
    random_center : bool, optional
        If True, a random center point will be selected. This overrides the center parameter.
    seed : int, optional
        Random seed for reproducibility when using random_center
        
    Returns:
    -------
    tuple
        (horizontal_acceleration, vertical_acceleration) - Two numpy arrays with values at grid nodes
    """
    if distribution != "uniform":
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Get grid dimensions
        rows, cols = grid.shape
        
        # Initialize arrays for node values
        num_nodes = grid.number_of_nodes
        horizontal_accel = np.zeros(num_nodes)
        vertical_accel = np.zeros(num_nodes)
        
        # Get x and y coordinates of all nodes
        node_x = grid.x_of_node
        node_y = grid.y_of_node
        
        # Create mask of valid (non-NaN) nodes
        valid_mask = ~np.isnan(node_x) & ~np.isnan(node_y)
        
        # Calculate center coordinates
        if random_center:
            # Get indices of valid nodes
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                # Fallback if no valid nodes (shouldn't happen)
                valid_indices = np.arange(num_nodes)
            
            # Select a random valid node as center
            center_node = np.random.choice(valid_indices)
            center_x = node_x[center_node]
            center_y = node_y[center_node]
            
            # Try to get row, col for reporting
            try:
                center_row, center_col = grid.node_row_col(center_node)
                print(f"Randomly selected center at row={center_row}, col={center_col} (node={center_node})")
            except:
                print(f"Randomly selected center at node={center_node}, x={center_x}, y={center_y}")
                
        elif center is None:
            # Use geometric center of valid nodes
            valid_x = node_x[valid_mask]
            valid_y = node_y[valid_mask]
            
            if len(valid_x) > 0:
                center_x = np.mean(valid_x)
                center_y = np.mean(valid_y)
            else:
                # Fallback to all nodes (unlikely scenario)
                center_x = np.mean(node_x[~np.isnan(node_x)])
                center_y = np.mean(node_y[~np.isnan(node_y)])
        else:
            # Center is provided as (row, col)
            center_row, center_col = center
            # Convert row, col to node ID
            center_row = min(max(0, center_row), rows - 1)
            center_col = min(max(0, center_col), cols - 1)
            center_node = grid.grid_coords_to_node_id(center_row, center_col)
            center_x = node_x[center_node]
            center_y = node_y[center_node]
        
        # Calculate normalization factors based on distribution
        # Use only valid nodes for distance calculations
        valid_x = node_x[valid_mask]
        valid_y = node_y[valid_mask]
        
        if distribution == "circular" or distribution == "exponential":
            # Euclidean distance
            distances = np.sqrt((valid_x - center_x)**2 + (valid_y - center_y)**2)
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
        elif distribution == "square":
            # Chebyshev distance (max of x/y distances)
            x_distances = np.abs(valid_x - center_x)
            y_distances = np.abs(valid_y - center_y)
            max_distance = max(np.max(x_distances) if len(x_distances) > 0 else 1.0, 
                              np.max(y_distances) if len(y_distances) > 0 else 1.0)
        elif distribution == "diamond":
            # Manhattan distance (sum of x/y distances)
            x_distances = np.abs(valid_x - center_x)
            y_distances = np.abs(valid_y - center_y)
            distances = x_distances + y_distances
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
        
        # Ensure max_distance is not zero
        max_distance = max(max_distance, 1e-10)
        
        # Calculate acceleration values for each node
        for i in range(num_nodes):
            # Skip nodes with NaN coordinates
            if np.isnan(node_x[i]) or np.isnan(node_y[i]):
                horizontal_accel[i] = 0.0
                vertical_accel[i] = 0.0
                continue
            
            # Calculate distance from center
            dx = node_x[i] - center_x
            dy = node_y[i] - center_y
            
            if distribution == "circular":
                distance = np.sqrt(dx**2 + dy**2)
                factor = max(0.0, 1.0 - distance / max_distance)
            elif distribution == "square":
                distance = max(abs(dx), abs(dy))
                factor = max(0.0, 1.0 - distance / max_distance)
            elif distribution == "diamond":
                distance = abs(dx) + abs(dy)  # Manhattan distance
                factor = max(0.0, 1.0 - distance / max_distance)
            elif distribution == "exponential":
                distance = np.sqrt(dx**2 + dy**2)
                # Scale factor for exponential decay
                decay_factor = 3.0  # Adjust for faster/slower decay
                factor = np.exp(-distance / (max_distance / decay_factor))
            
            # Set the acceleration values
            horizontal_accel[i] = max_horizontal * factor
            vertical_accel[i] = max_vertical * factor
    else:
        horizontal_accel = np.ones_like(grid.at_node["topographic__elevation"])
        vertical_accel = np.ones_like(grid.at_node["topographic__elevation"])
        
        horizontal_accel[np.isnan(grid.at_node["topographic__elevation"])] = np.nan
        vertical_accel[np.isnan(grid.at_node["topographic__elevation"])] = np.nan
        
        horizontal_accel[grid.core_nodes] *= max_horizontal
        vertical_accel[grid.core_nodes] *= max_vertical
    
    print(f"{distribution} horizontal and vertical PGA arrays generated")
    return horizontal_accel, vertical_accel
# %%% Calculate Factor of Safety - Infinite slope

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

# %%% Factor of safety

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

# %%% Calculate critical acceleration

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

# %%% Calculate critical transient acceleration

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

# %% Identify landslide paths
# %%% Calculate Newmark displacement

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

# %%% Trace landslide paths
def trace_paths_landslides(grid, starting_nodes, newmark_distances,
                           check_sum=False, print_path_details=False):
    """
    Calculates all downslope paths taken from a set of starting nodes constrained 
    by the newmark distance of each node and assigns proportions for each path

    Parameters
    ----------
    grid : Landlab RasterModelGrid
        DESCRIPTION.
    starting_nodes : numpy.ndarray
        1D array of node ids to trace paths for.
    max_distances : numpy.ndarray
        2D array of same size as the landlab grid containing the newmark distances for each node.
    check_sum : bool, optional
        Checks the sum of final proportions for all paths from a each node.
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
    receiver_nodes = grid.at_node['hill_flow__receiver_node']
    receiver_proportions = grid.at_node['hill_flow__receiver_proportions']
    boundary_nodes = set(grid.boundary_nodes)  # Convert to set for faster lookup
    
    final_nodes = []
    final_proportions = []
    path_details = {}  # To store details of each path for each starting node
    
    for node in tqdm(starting_nodes):
        newmark_distance = newmark_distances[node]
        stack = [(0.0, node, 1.0, [node])]  # (current_distance, current_node, current_proportion, path)
        path_details[node] = []  # Initialize path details for this starting node

        while stack:
            current_distance, current_node, current_proportion, path = heapq.heappop(stack)

            if current_distance >= newmark_distance:
                
                final_nodes.append((node, current_node))
                final_proportions.append(current_proportion)
                path_details[node].append((path, current_proportion))
                continue
            
            receivers = receiver_nodes[current_node]
            receiver_props = receiver_proportions[current_node]
            
            if receivers[0] == current_node:
                # print('a')
                break
                # print('b')
            else: 
                for receiver, prop in zip(receivers, receiver_props):
                        if receiver != -1:
                            new_distance = current_distance + calculate_node_distance(grid, current_node, receiver)
                            new_path = path + [receiver]
                            new_proportion = current_proportion * prop
        
                            # Check if the receiver node is a boundary node
                            if receiver in boundary_nodes or grid.at_node['topographic__elevation'][receiver] == np.nan:
                                final_nodes.append((node, current_node))
                                final_proportions.append(current_proportion)
                                path_details[node].append((path, current_proportion))
                            else:
                                heapq.heappush(stack, (new_distance, receiver, new_proportion, new_path))
    
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
                # print(f"Proportions for node {start_node} do not sum to one: {total_proportion}")
                
        return final_nodes, final_proportions, path_details, proportion_not_zero
    
    else:
        return final_nodes, final_proportions, path_details

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
    
# %%% Update soil depth
def update_soil_depth(paths, proportions, soil_depth):
    """
    Uses outputs from trace_paths_landslides to calculate change in soil depth

    Parameters
    ----------
    paths : list of tuples
        Lists each path as a tuple of node_ids, starting with the initial node
    proportions : list
        List containing the final proportions for each path defined in final_nodes
    soil_depth : numpy.ndarray
        1D array of floats representing the depth of soil in meters at each node

    Returns
    -------
    updated_soil_depth : numpy.ndarray
        1D array of floats representing the new soil depth in meters at each node
    soil_deposition : numpy.ndarray
        1D array of floats representing the amount of soil **deposited** at each node
    soil_erosion : numpy.ndarray
        1D array of floats representing the amount of soil **eroded** from each node

    """
    updated_soil_depth = soil_depth.copy()
    
    for path, proportion in tqdm(zip(paths, proportions)):
        for i in range(len(path) - 1):
            start_node = path[i]
            end_node = path[i + 1]
            soil_movement = updated_soil_depth[start_node] * proportion
            if updated_soil_depth[start_node] - soil_movement < 0:
                soil_movement = updated_soil_depth[start_node]  # Move only the available soil
            # Soil erosion
            updated_soil_depth[start_node] -= soil_movement
            # Soil deposition
            updated_soil_depth[end_node] += soil_movement
    
    return updated_soil_depth

# %% Statistical functions

def fit_bivariate_kde(dataframe, x_col, y_col, category_col=None, 
                                     log_transform=True, bandwidth=None, n_levels=20, 
                                     cmap="viridis", figsize=(12, 10), plot_results=True):
    """
    Create bivariate KDEs for overall data and by category, and return KDE objects for sampling.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing the bivariate data
    x_col, y_col : str
        Names of the columns for the variables
    category_col : str or None
        Name of the categorical column to group by (None for no grouping)
    log_transform : bool or tuple
        If/which variables to log-transform (True for both, or tuple for selective)
    bandwidth : float, array-like, or dict
        Bandwidth for the KDE (can be different per group if dict)
    n_levels : int
        Number of contour levels for the plot
    cmap : str
        Colormap for the contour plot
    figsize : tuple
        Figure size for plots
    plot_results : bool
        Whether to generate plots (set to False to only return KDE objects)
    
    Returns:
    --------
    kde_results : dict
        Dictionary with 'overall' KDE and group-specific KDEs if categorical variable provided
    transform_info : dict
        Information about the transformation for use in sampling
    """
    # Make a copy to avoid modifying the original dataframe
    x_vals = np.array(dataframe[x_col])
    y_vals = np.array(dataframe[y_col])
    
    # Set up transformation info
    transform_info = {
        'x_col': x_col,
        'y_col': y_col,
        'log_x': False,
        'log_y': False,
    }
    
    # Handle log transformation
    if isinstance(log_transform, tuple):
        log_x, log_y = log_transform
    else:
        log_x = log_y = log_transform
    
    # Apply transformations if requested
    if log_x:
        if x_vals.min() <= 0:
            raise ValueError(f"Cannot log-transform {x_col}: contains values <= 0")
        x_vals = np.log(x_vals)
        transform_info['log_x'] = True
    
    if log_y:
        if y_vals.min() <= 0:
            raise ValueError(f"Cannot log-transform {y_col}: contains values <= 0")
        y_vals = np.log(y_vals)
        transform_info['log_y'] = True
    
    # Calculate appropriate bounds for each variable
    # These prevent KDE from extending to unreasonable regions
    if log_x:
        x_min = x_vals.min() - 0.1 * np.abs(x_vals.min())
        x_max = x_vals.max() + 0.1 * np.abs(x_vals.max())
    else:
        x_std = x_vals.std()
        x_min = x_vals.min() - 0.5 * x_std
        x_max = x_vals.max() + 0.5 * x_std
    
    if log_y:
        y_min = y_vals.min() - 0.1 * np.abs(y_vals.min())
        y_max = y_vals.max() + 0.1 * np.abs(y_vals.max())
    else:
        y_std = y_vals.std()
        y_min = y_vals.min() - 0.5 * y_std
        y_max = y_vals.max() + 0.5 * y_std
    
    # Store bounds information
    x_bounds = (x_min, x_max)
    y_bounds = (y_min, y_max)
    transform_info['x_bounds'] = x_bounds
    transform_info['y_bounds'] = y_bounds
    
    # Initialize results dictionary
    kde_results = {}
    
    # Create overall KDE
    data = np.vstack([x_vals, y_vals])
    kde_overall = stats.gaussian_kde(data, bw_method=bandwidth)
    kde_results['overall'] = kde_overall
    
    # Group data by category if provided
    if category_col is not None and category_col in dataframe.columns:
        # Get unique categories
        categories = dataframe[category_col].unique()
        transform_info['categories'] = list(categories)
        
        # Create dictionary to store KDEs by category
        kde_by_category = {}
        
        # Create KDE for each category
        for category in categories:
            category_mask = dataframe[category_col] == category
            
            # Skip if too few data points
            if np.sum(category_mask) < 5:
                print(f"Warning: Category '{category}' has fewer than 5 data points. Skipping KDE.")
                continue
                
            cat_x_vals = x_vals[category_mask] if isinstance(x_vals, np.ndarray) else np.array(x_vals[category_mask])
            cat_y_vals = y_vals[category_mask] if isinstance(y_vals, np.ndarray) else np.array(y_vals[category_mask])
            
            cat_data = np.vstack([cat_x_vals, cat_y_vals])
            
            # Get bandwidth for this category
            cat_bandwidth = bandwidth
            if isinstance(bandwidth, dict):
                cat_bandwidth = bandwidth.get(category, None)
                
            # Create KDE for this category
            try:
                cat_kde = stats.gaussian_kde(cat_data, bw_method=cat_bandwidth)
                kde_by_category[category] = cat_kde
            except np.linalg.LinAlgError:
                print(f"Warning: Could not create KDE for category '{category}'. Insufficient or collinear data.")
                continue
        
        # Add category KDEs to results
        kde_results['by_category'] = kde_by_category
    
    # Generate plots if requested
    if plot_results:
        plot_bivariate_kde(dataframe, x_col, y_col, category_col, 
                           kde_results, transform_info, 
                           n_levels=n_levels, cmap=cmap, figsize=figsize)
    
    # Return the KDEs and transformation info
    return kde_results, transform_info

def plot_bivariate_kde(dataframe, x_col, y_col, category_col=None, 
                      kde_results=None, transform_info=None, 
                      n_levels=20, cmap="viridis", figsize=(12, 10)):
    """
    Plot bivariate KDEs for overall data and by category.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing the bivariate data
    x_col, y_col : str
        Names of the columns for the variables
    category_col : str or None
        Name of the categorical column used for grouping
    kde_results : dict
        Dictionary with KDE objects returned by fit_bivariate_kde_with_categories
    transform_info : dict
        Transformation information from fit_bivariate_kde_with_categories
    n_levels : int
        Number of contour levels for the plot
    cmap : str
        Colormap for the contour plot
    figsize : tuple
        Figure size for plots
    """
    if kde_results is None or transform_info is None:
        # Call the main function to get KDE results if not provided
        kde_results, transform_info = fit_bivariate_kde(
            dataframe, x_col, y_col, category_col, plot_results=False)
    
    # Extract transformation info
    log_x = transform_info.get('log_x', False)
    log_y = transform_info.get('log_y', False)
    x_bounds = transform_info.get('x_bounds')
    y_bounds = transform_info.get('y_bounds')
    
    # Create transformed data for plotting
    x_vals = np.array(dataframe[x_col])
    y_vals = np.array(dataframe[y_col])
    
    if log_x:
        x_vals = np.log(x_vals)
    if log_y:
        y_vals = np.log(y_vals)
    
    # Create grid for evaluation
    n_grid = 100
    x_grid = np.linspace(x_bounds[0], x_bounds[1], n_grid)
    y_grid = np.linspace(y_bounds[0], y_bounds[1], n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Plot the overall KDE in transformed space
    plt.figure(figsize=figsize)
    
    # Plot the transformed data points
    plt.scatter(x_vals, y_vals, alpha=0.3, s=10, color='black')
    
    # Evaluate and plot overall KDE
    kde_overall = kde_results['overall']
    Z = kde_overall(positions).reshape(X.shape)
    
    # Plot the contours
    contour = plt.contourf(X, Y, Z, levels=n_levels, cmap=cmap, alpha=0.8)
    plt.colorbar(contour, label='Density')
    
    # Add contour lines
    contour_lines = plt.contour(X, Y, Z, levels=n_levels, colors='white', linewidths=0.5, alpha=0.5)
    
    # Set axis labels based on transformation
    x_label = f"log({x_col})" if log_x else x_col
    y_label = f"log({y_col})" if log_y else y_col
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.title(f"Overall Bivariate KDE in Transformed Space")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Also show original scale plot with seaborn
    plt.figure(figsize=figsize)
    
    # Create a joint plot for the original data
    g = sns.jointplot(
        data=dataframe,
        x=x_col,
        y=y_col,
        kind="scatter",
        height=8,
        alpha=0.4,
    )
    
    # Set log scales if needed
    if log_x:
        g.ax_joint.set_xscale('log')
        g.ax_marg_x.set_xscale('log')
    
    if log_y:
        g.ax_joint.set_yscale('log')
        g.ax_marg_y.set_yscale('log')
    
    # Add KDE plots to the margins
    if log_x:
        sns.kdeplot(x=x_vals, ax=g.ax_marg_x, log_scale=True, color='blue', fill=True)
    else:
        sns.kdeplot(x=x_vals, ax=g.ax_marg_x, color='blue', fill=True)
        
    if log_y:
        sns.kdeplot(y=y_vals, ax=g.ax_marg_y, log_scale=True, color='blue', fill=True)
    else:
        sns.kdeplot(y=y_vals, ax=g.ax_marg_y, color='blue', fill=True)
    
    plt.suptitle(f"Joint Distribution of {x_col} and {y_col} (Original Scale)", y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Plot KDEs by category if available
    if category_col is not None and 'by_category' in kde_results:
        # Create a color map for categories
        categories = transform_info.get('categories', list(kde_results['by_category'].keys()))
        n_categories = len(categories)
        
        if n_categories > 0:
            # Create a categorical plot first
            plt.figure(figsize=figsize)
            
            # Get a colormap with distinct colors
            category_cmap = plt.get_cmap(cmap, n_categories)
            colors = [category_cmap(i/n_categories) for i in range(n_categories)]
            
            # Create scatter plot by category
            for i, category in enumerate(categories):
                category_mask = dataframe[category_col] == category
                
                if category in kde_results['by_category']:
                    cat_x = x_vals[category_mask] if isinstance(x_vals, np.ndarray) else np.array(x_vals[category_mask])
                    cat_y = y_vals[category_mask] if isinstance(y_vals, np.ndarray) else np.array(y_vals[category_mask])
                    
                    plt.scatter(cat_x, cat_y, alpha=0.5, s=20, color=colors[i], label=category)
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"Data Points by {category_col}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Plot KDE for each category
            plt.figure(figsize=figsize)
            
            # Plot the data points in background with low alpha
            plt.scatter(x_vals, y_vals, alpha=0.1, s=5, color='gray')
            
            # Plot contours for each category
            for i, category in enumerate(categories):
                if category in kde_results['by_category']:
                    cat_kde = kde_results['by_category'][category]
                    Z = cat_kde(positions).reshape(X.shape)
                    
                    # Plot the contour lines for this category
                    contour_lines = plt.contour(X, Y, Z, levels=int(n_levels/2), 
                                               colors=[colors[i]], linewidths=1.5, 
                                               alpha=0.8, label=f"{category}")
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"Bivariate KDE Contours by {category_col}")
            
            # Create proxy artists for the legend
            legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=2, label=cat) 
                              for i, cat in enumerate(categories) if cat in kde_results['by_category']]
            plt.legend(handles=legend_elements)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
            # Create separate KDE plots for each category
            fig, axes = plt.subplots(int(np.ceil(n_categories/2)), 2, figsize=figsize)
            axes = axes.flatten() if n_categories > 1 else [axes]
            
            for i, category in enumerate(categories):
                if i < len(axes) and category in kde_results['by_category']:
                    ax = axes[i]
                    
                    # Get category data
                    category_mask = dataframe[category_col] == category
                    cat_x = x_vals[category_mask] if isinstance(x_vals, np.ndarray) else np.array(x_vals[category_mask])
                    cat_y = y_vals[category_mask] if isinstance(y_vals, np.ndarray) else np.array(y_vals[category_mask])
                    
                    # Plot data points
                    ax.scatter(cat_x, cat_y, alpha=0.4, s=10, color=colors[i])
                    
                    # Evaluate and plot KDE
                    cat_kde = kde_results['by_category'][category]
                    Z = cat_kde(positions).reshape(X.shape)
                    contour = ax.contourf(X, Y, Z, levels=n_levels, cmap=plt.get_cmap('viridis'), alpha=0.3)
                    contour_lines = ax.contour(X, Y, Z, levels=int(n_levels/2), colors=['black'], linewidths=0.5)
                    
                    ax.set_title(f"{category} (n={np.sum(category_mask)})")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.grid(True, alpha=0.3)
            
            # Hide any unused axes
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
                
            plt.suptitle(f"Individual KDEs by {category_col}", y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()

def conditional_sample(kde_results, transform_info, condition_var, condition_value, 
                      n_samples=1000, tolerance=0.1, category=None):
    """
    Sample from a conditional distribution with proper bounds handling.
    
    Parameters:
    -----------
    kde_results : dict or scipy.stats.gaussian_kde
        Kernel density estimator results from fit_bivariate_kde_with_categories
        or a direct kde object
    transform_info : dict
        Information about transformations and bounds
    condition_var : str
        Name of the conditioning variable
    condition_value : float
        Value of the conditioning variable (in original scale)
    n_samples : int
        Number of samples to generate
    tolerance : float
        Tolerance for accepting samples
    category : str or None
        Category to sample from (None for overall KDE)
    
    Returns:
    --------
    samples : numpy.ndarray
        Array of samples (in original scale)
    """
    # Get the appropriate KDE
    if isinstance(kde_results, dict):
        if category is not None and 'by_category' in kde_results and category in kde_results['by_category']:
            kde = kde_results['by_category'][category]
        else:
            kde = kde_results['overall']
    else:
        # If kde_results is already a KDE object, use it directly
        kde = kde_results
    
    # Determine which variable is the conditioning one
    if condition_var == transform_info['x_col']:
        condition_idx = 0
        return_idx = 1
        log_condition = transform_info['log_x']
        log_return = transform_info['log_y']
        return_bounds = transform_info['y_bounds']
    else:
        condition_idx = 1
        return_idx = 0
        log_condition = transform_info['log_y']
        log_return = transform_info['log_x']
        return_bounds = transform_info['x_bounds']
    
    # Transform the condition value if necessary
    if log_condition:
        if condition_value <= 0:
            raise ValueError(f"Cannot log-transform condition value {condition_value}: must be > 0")
        transformed_condition_value = np.log(condition_value)
    else:
        transformed_condition_value = condition_value
    
    # Generate samples using rejection sampling
    max_attempts = n_samples * 100
    batch_size = min(10000, n_samples * 5)
    accepted_samples = []
    attempts = 0
    
    while len(accepted_samples) < n_samples and attempts < max_attempts:
        # Generate a batch of samples
        samples_batch = kde.resample(batch_size)
        
        # Find samples where conditioning variable is close to target
        condition_samples = samples_batch[condition_idx, :]
        mask = np.abs(condition_samples - transformed_condition_value) < tolerance
        
        # Apply bounds to response variable
        valid_responses = samples_batch[return_idx, mask]
        if len(valid_responses) > 0:
            bounds_mask = (valid_responses >= return_bounds[0]) & (valid_responses <= return_bounds[1])
            valid_responses = valid_responses[bounds_mask]
            accepted_samples.extend(valid_responses)
        
        attempts += batch_size
    
    # If we don't have enough samples, use the ones we have
    if len(accepted_samples) < n_samples:
        print(f"Warning: Only obtained {len(accepted_samples)} valid samples instead of {n_samples}")
        if len(accepted_samples) == 0:
            # If no samples, fall back to direct sampling from marginal
            print("Falling back to sampling from marginal distribution")
            samples_batch = kde.resample(n_samples)
            accepted_samples = samples_batch[return_idx, :]
    else:
        # Trim to the requested number
        accepted_samples = accepted_samples[:n_samples]
    
    # Convert back to numpy array
    result_samples = np.array(accepted_samples)
    
    # Transform back to original scale if necessary
    if log_return:
        result_samples = np.exp(result_samples)
    
    return result_samples

def plot_conditional_samples(dataframe, x_col, y_col, kde_results, transform_info, 
                           condition_var, condition_values, n_samples=100, 
                           figsize=(12, 8), category=None, category_col=None):
    """
    Plot conditional samples for multiple condition values.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        Original dataframe
    x_col, y_col : str
        Column names
    kde_results : dict or scipy.stats.gaussian_kde
        The fitted KDE results or direct KDE object
    transform_info : dict
        Transformation information
    condition_var : str
        Variable to condition on
    condition_values : list
        Values to condition on
    n_samples : int
        Number of samples per condition
    figsize : tuple
        Figure size
    category : str or None
        Category to use for sampling (None for overall)
    category_col : str or None
        Name of the category column for filtering original data
    """
    plt.figure(figsize=figsize)
    
    # Filter original data if category is specified
    if category is not None and category_col is not None:
        plot_data = dataframe[dataframe[category_col] == category].copy()
        title_suffix = f" (Category: {category})"
    else:
        plot_data = dataframe.copy()
        title_suffix = ""
    
    # Plot original data
    plt.scatter(plot_data[x_col], plot_data[y_col], alpha=0.2, color='gray', label='Original data')
    
    # Determine if we're conditioning on X or Y
    is_x_condition = (condition_var == x_col)
    
    # Sample and plot for each condition value
    colors = plt.cm.tab10(np.linspace(0, 1, len(condition_values)))
    
    for i, value in enumerate(condition_values):
        samples = conditional_sample(kde_results, transform_info, condition_var, value, 
                                   n_samples, category=category)
        
        if is_x_condition:
            # Conditioning on X, samples are Y values
            plt.scatter([value] * len(samples), samples, alpha=0.7, color=colors[i], 
                      label=f"{condition_var}={value:.2f}")
        else:
            # Conditioning on Y, samples are X values
            plt.scatter(samples, [value] * len(samples), alpha=0.7, color=colors[i],
                      label=f"{condition_var}={value:.2f}")
    
    # Set axis scales if needed
    if transform_info['log_x']:
        plt.xscale('log')
    if transform_info['log_y']:
        plt.yscale('log')
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"Conditional Sampling: P({'Y' if is_x_condition else 'X'}|{condition_var}){title_suffix}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_multiple_conditional_samples(dataframe, x_col, y_col, kde_results, transform_info, 
                                    condition_var, condition_values, n_samples=100, 
                                    figsize=(15, 10), category_col=None):
    """
    Plot conditional samples for both overall and category-specific KDEs.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        Original dataframe
    x_col, y_col : str
        Column names
    kde_results : dict
        The fitted KDE results
    transform_info : dict
        Transformation information
    condition_var : str
        Variable to condition on
    condition_values : list
        Values to condition on
    n_samples : int
        Number of samples per condition
    figsize : tuple
        Figure size
    category_col : str or None
        Name of the category column
    """
    # First plot the overall conditional samples
    plot_conditional_samples(dataframe, x_col, y_col, kde_results, transform_info,
                           condition_var, condition_values, n_samples, figsize)
    
    # If we have categories, plot conditional samples for each category
    if category_col is not None and 'by_category' in kde_results:
        categories = transform_info.get('categories', list(kde_results['by_category'].keys()))
        
        for category in categories:
            if category in kde_results['by_category']:
                plot_conditional_samples(dataframe, x_col, y_col, kde_results, transform_info,
                                       condition_var, condition_values, n_samples, figsize,
                                       category=category, category_col=category_col)
