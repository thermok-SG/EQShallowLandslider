

"""
Extra functions for analysing the topography

Contains functions for:
    - Plotting rose diagrams (polar histograms) for aspect
    - Excess topography (currently not working)

"""
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import vonmises
import matplotlib.pyplot as plt
import richdem as rd

from scipy.ndimage import (
    grey_erosion, grey_dilation,
    gaussian_filter
)

def calculate_terrain_attribute(grid, field_name, attrib, out_field=None):
    """
    Compute a terrain attribute with richdem and add it to a Landlab grid.
    Automatically detects nodata values from the field.

    Parameters
    ----------
    mg : landlab.ModelGrid
        The Landlab grid (must be RasterModelGrid for reshaping).
    field_name : str
        Name of the Landlab field containing elevation (e.g. 'topographic__elevation').
    attrib : str
        Richdem terrain attribute to calculate (e.g. 'slope_riserun', 'curvature').
    out_field : str, optional
        Name of the output field in Landlab (defaults to attrib name).

    Returns
    -------
    numpy.ndarray
        The computed attribute values (1D, node order).
    """
    if out_field is None:
        out_field = attrib

    # Grab field from landlab
    z = grid.at_node[field_name]
    nrows, ncols = grid.shape
    dem2d = z.reshape((nrows, ncols))

    # Detect nodata: if any NaNs, set nodata to np.nan, else use -9999
    if np.isnan(dem2d).any():
        nodata = np.nan
    else:
        nodata = -9999  # fallback

    # Wrap into richdem rdarray
    dem_rd = rd.rdarray(dem2d.copy(), no_data=nodata)
    dem_rd.geotransform = [0, grid.dx, 0, 0, 0, -grid.dy]

    # Compute terrain attribute with richdem
    result2d = rd.TerrainAttribute(dem_rd, attrib=attrib)

    # Flatten to Landlab node order
    result1d = np.asarray(result2d).ravel()

    # Add to Landlab grid
    grid.add_field(out_field, result1d, at='node', clobber=True)

    return result1d

# -------------------------------
# Helper: Rose overlay
# -------------------------------
def _plot_rose_overlay(datasets, labels, colors, processed_data, bin_centers, width, global_max,
                        normalize, log_scale):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, layout='constrained')
    for dataset, hist, label, color in zip(datasets, processed_data, labels, colors):
        ax.bar(bin_centers, hist, width=width, bottom=0.0,
                color=color, edgecolor='black', alpha=0.5,
                label=f"{label} (n={len(dataset)})")
    ax.set_ylim(0, global_max * 1.1)
    ax.set_title(f"{'Log10 ' if log_scale else ''}{'Normalized ' if normalize else ''}Rose Diagram")
    ax.legend(loc="best", bbox_to_anchor=(1.3, 0))
    _set_polar_ticks(ax)


# -------------------------------
# Helper: Rose subplots
# -------------------------------
def _plot_rose_subplots(datasets, labels, colors, processed_data, bin_centers, width, global_max):
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                            figsize=(min(15, n_cols * 5), n_rows * 4),
                            subplot_kw={'projection': 'polar'},
                            layout='constrained')
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (dataset, hist, label, color) in enumerate(zip(datasets, processed_data, labels, colors)):
        ax = axes[i]
        ax.bar(bin_centers, hist, width=width, bottom=0.0,
                color=color, edgecolor='black', alpha=0.7)
        ax.set_ylim(0, global_max * 1.1)
        ax.set_title(f"{label}\n(n={len(dataset)})", pad=20)
        _set_polar_ticks(ax)

    for i in range(n_datasets, len(axes)):
        axes[i].set_visible(False)


# -------------------------------
# Helper: KDE curves
# -------------------------------
def _plot_kde(datasets_rad, labels, colors, kde_kappa, kde_points):
    theta = np.linspace(0, 2*np.pi, kde_points)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, layout='constrained')
    for data, label, color in zip(datasets_rad, labels, colors):
        pdf_vals = np.zeros_like(theta)
        for angle in data:
            pdf_vals += vonmises.pdf(theta, kde_kappa, loc=angle)
        pdf_vals /= pdf_vals.max()  # normalize for comparability
        ax.plot(theta, pdf_vals, color=color, lw=2, label=f"{label} (n={len(data)})")

    ax.set_title("Circular KDE of Topographic Aspect")
    ax.legend(loc="best", bbox_to_anchor=(1.3, 0))
    _set_polar_ticks(ax)


# -------------------------------
# Helper: Polar ticks setup
# -------------------------------
def _set_polar_ticks(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    tick_positions = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ax.set_thetagrids(np.degrees(tick_positions),
                    ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])


# -------------------------------
# Main dispatcher
# -------------------------------
def plot_aspect(
    datasets,
    labels=None,
    colors=None,
    normalize=True,
    log_scale=False,
    mode="rose",        # "rose" or "kde"
    arrangement="overlay",  # only for rose: "overlay" or "subplots"
    n_bins=16,
    kde_kappa=20,
    kde_points=360
):
    """
    Plot topographic aspect data as rose diagrams or circular KDE curves.
    """
    # Ensure datasets are numpy arrays and wrap to 0–360
    datasets = [np.array(d) for d in datasets]
    datasets = [d[d >= 0] % 360 for d in datasets]
    print("Plotting aspect datasets...")

    # Default labels and colors
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(datasets))]
    if colors is None:
        colors = ['skyblue', 'lightgreen', 'salmon', 'purple', 'orange']

    if mode == "rose":
        # Precompute histograms
        processed_data = []
        global_max = 0
        for dataset in datasets:
            hist, _ = np.histogram(dataset, bins=n_bins, range=(0, 360))
            hist = np.maximum(0, hist)
            if normalize:
                hist = (hist / len(dataset)) * 100
            if log_scale:
                hist = np.log10(hist + 1)
            processed_data.append(hist)
            global_max = max(global_max, np.max(hist))

        # Bin geometry
        bin_edges = np.linspace(0, 360, n_bins + 1)
        bin_centers = np.deg2rad(bin_edges[:-1] + np.diff(bin_edges)/2)
        width = np.deg2rad(360 / n_bins)

        if arrangement == "overlay":
            _plot_rose_overlay(datasets, labels, colors, processed_data,
                                bin_centers, width, global_max, normalize, log_scale)
        elif arrangement == "subplots":
            _plot_rose_subplots(datasets, labels, colors, processed_data,
                                bin_centers, width, global_max)
        else:
            raise ValueError("arrangement must be 'overlay' or 'subplots'")

    elif mode == "kde":
        datasets_rad = [np.deg2rad(d) for d in datasets]
        _plot_kde(datasets_rad, labels, colors, kde_kappa, kde_points)

    else:
        raise ValueError("mode must be 'rose' or 'kde'")

    plt.show()


def plot_aspect_roses_older(datasets, labels=None, colors=None, normalize=True, log_scale=False):
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
    datasets = [np.array(dataset) for dataset in datasets]
    datasets = [dataset[dataset >= 0] % 360 for dataset in datasets]
    print('Plotting aspects...')

    # Default labels and colors if not provided
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(datasets))]
    if colors is None:
        colors = ['skyblue', 'lightgreen', 'salmon', 'purple', 'orange']

    # Set up the plot
    plt.figure(layout='constrained')
    ax = plt.subplot(111, projection='polar')

    # Number of bins
    n_bins = 16
    max_percentage = 0

    # Plot each dataset
    for i, (dataset, label_num, color) in enumerate(zip(datasets, labels, colors)):
        # Create histogram
        hist, bin_edges = np.histogram(dataset, bins=n_bins, range=(0, 360))
        hist = np.maximum(0, hist)

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

    # Set tick positions and labels for cardinal directions
    tick_positions = np.linspace(0, 2*np.pi, 8, endpoint=False)
    ax.set_thetagrids(np.degrees(tick_positions), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    
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
    plt.legend(loc='best', bbox_to_anchor=(1.3, 0))

    # plt.tight_layout()
    plt.show()
    
def plot_aspect_roses_old(datasets, labels=None, colors=None, normalize=True, log_scale=False):
    """
    Create separate rose diagrams for each dataset in subplots.
    
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
    datasets = [np.array(dataset) for dataset in datasets]
    datasets = [dataset[dataset >= 0] % 360 for dataset in datasets]
    print('Plotting aspects...')
    
    # Default labels and colors if not provided
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(datasets))]
    if colors is None:
        colors = ['skyblue', 'lightgreen', 'salmon', 'purple', 'orange']
    
    # Calculate subplot layout
    n_datasets = len(datasets)
    n_cols = min(3, n_datasets)  # Maximum 3 columns
    n_rows = (n_datasets + n_cols - 1) // n_cols  # Ceiling division
    
    # Set up the figure - adjust size based on number of subplots
    fig_width = min(15, n_cols * 5)  # Max width of 15, scale with columns
    fig_height = n_rows * 4  # 4 inches per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), 
                            subplot_kw={'projection': 'polar'}, 
                            layout='constrained')
    
    # Handle different subplot configurations
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten to 1D array for easier indexing
    
    # Number of bins
    n_bins = 16
    
    # Find global max for consistent scaling across subplots
    global_max = 0
    processed_data = []
    
    for dataset in datasets:
        hist, _ = np.histogram(dataset, bins=n_bins, range=(0, 360))
        hist = np.maximum(0, hist)
        if normalize:
            hist = (hist / len(dataset)) * 100
        if log_scale:
            hist = np.log10(hist + 1)
        processed_data.append(hist)
        global_max = max(global_max, np.max(hist))
    
    # Plot each dataset in its own subplot
    for i, (dataset, hist, label_num, color) in enumerate(zip(datasets, processed_data, labels, colors)):
        ax = axes[i]  # Simple indexing with flattened array
        
        # Calculate bin centers and width
        bin_edges = np.linspace(0, 360, n_bins + 1)
        bin_centers = np.deg2rad(bin_edges[:-1] + np.diff(bin_edges)/2)
        width = np.deg2rad(360 / n_bins)
        
        # Plot the rose diagram
        ax.bar(bin_centers, hist, width=width, bottom=0.0,
                color=color, edgecolor='black', alpha=0.7)
        
        # Customize each subplot
        ax.set_theta_zero_location('N')  # 0 degrees at the top
        ax.set_theta_direction(-1)  # Clockwise
        
        # Set tick positions and labels for cardinal directions
        tick_positions = np.linspace(0, 2*np.pi, 8, endpoint=False)
        ax.set_thetagrids(
            np.degrees(tick_positions),
                        ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
                        )
        
        # Set consistent radial limits for all subplots
        ax.set_ylim(0, global_max * 1.1)
        
        # Set subplot title
        scale_type = "Log10 " if log_scale else ""
        norm_type = "Normalized " if normalize else ""
        ax.set_title(f'{label_num}\n(n={len(dataset)})', pad=20)
        
        # Customize radial labels based on scale type
        if log_scale:
            max_tick = min(2, int(global_max * 1.1))
            ax.set_rticks(np.linspace(0, max_tick, 5))
    
    # Hide any unused subplots
    for i in range(n_datasets, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title
    scale_type = "Log10 " if log_scale else ""
    norm_type = "Normalized " if normalize else ""
    # fig.suptitle(f'{scale_type}{norm_type}Topographic Aspect Rose Diagrams', fontsize=16)
    
    plt.show()

# %% Function to calculate excess topography based on TopoToolbox method [currently not working]
def calculate_excess_topography(grid, method='planar', **kwargs):
    """
    Calculate excess topography for a Landlab grid.
    
    Parameters:
    -----------
    grid : RasterModelGrid
        Landlab grid with topographic__elevation field
    method : str
        Method for calculating reference surface:
        - 'planar': Fit a plane through the topography
        - 'polynomial': Fit polynomial surface
        - 'channel_based': Use channel network to define base level
        - 'valley_based': Interpolate from valley bottoms
    **kwargs : additional arguments for specific methods
    
    Returns:
    --------
    excess_topo : numpy array
        Excess topography values at each node
    """
    
    # Get elevation data
    z = grid.at_node['topographic__elevation']
    x, y = grid.x_of_node, grid.y_of_node
    
    if method == 'planar':
        return _planar_reference(x, y, z, **kwargs)
    elif method == 'polynomial':
        return _polynomial_reference(x, y, z, **kwargs)
    elif method == 'channel_based':
        return _channel_based_reference(grid, **kwargs)
    elif method == 'valley_based':
        return _valley_based_reference(grid, **kwargs)
    elif method == 'morphological':
        return _morphological_reference(grid, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

# %%% Excess topo - helper functions
def _planar_reference(x, y, z, min_elevation_percentile=10):
    """
    Calculate excess topography using a planar reference surface.
    """
    # Remove NaN values
    valid_mask = ~np.isnan(z)
    x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]
    
    # Optionally use only lower elevations to define the plane
    if min_elevation_percentile > 0:
        threshold = np.percentile(z_valid, min_elevation_percentile)
        low_mask = z_valid <= threshold
        x_fit, y_fit, z_fit = x_valid[low_mask], y_valid[low_mask], z_valid[low_mask]
    else:
        x_fit, y_fit, z_fit = x_valid, y_valid, z_valid
    
    # Fit plane: z = ax + by + c
    A = np.column_stack([x_fit, y_fit, np.ones(len(x_fit))])
    coeffs, _, _, _ = np.linalg.lstsq(A, z_fit, rcond=None)
    
    # Calculate reference surface for all points
    z_ref = coeffs[0] * x + coeffs[1] * y + coeffs[2]
    
    # Calculate excess topography
    excess = z - z_ref
    excess[excess < 0] = 0  # Only positive excess
    
    return excess

def _polynomial_reference(x, y, z, degree=2):
    """
    Calculate excess topography using polynomial reference surface.
    """
    valid_mask = ~np.isnan(z)
    x_valid, y_valid, z_valid = x[valid_mask], y[valid_mask], z[valid_mask]
    
    # Create polynomial terms
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x_valid**i) * (y_valid**j))
    
    A = np.column_stack(terms)
    coeffs, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
    
    # Calculate reference surface for all points
    z_ref = np.zeros_like(z)
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            z_ref += coeffs[idx] * (x**i) * (y**j)
            idx += 1
    
    excess = z - z_ref
    excess[excess < 0] = 0
    
    return excess

def _channel_based_reference(grid, min_drainage_area=1000, outlet_method='min_elevation'):
    """
    Calculate excess topography using channel network as reference.
    Assumes flow routing already done with PriorityFloodFlowRouter.
    """
    # Use existing drainage area field (should be 'drainage_area' or 'surface_water__discharge')
    if 'drainage_area' in grid.at_node:
        drainage_area = grid.at_node['drainage_area']
    elif 'surface_water__discharge' in grid.at_node:
        drainage_area = grid.at_node['surface_water__discharge']
    else:
        raise ValueError("No drainage area field found. Expected 'drainage_area' or 'surface_water__discharge'")
    
    # Get drainage area (convert to actual area if needed)
    if np.max(drainage_area) < 1:  # Likely in units of grid cells
        drainage_area = drainage_area * grid.dx * grid.dy
    
    # Define channel network
    channel_mask = drainage_area > min_drainage_area
    channel_nodes = np.where(channel_mask)[0]
    
    if len(channel_nodes) == 0:
        raise ValueError("No channels found with given drainage area threshold")
    
    # Get channel elevations
    channel_elevations = grid.at_node['topographic__elevation'][channel_nodes]
    channel_x = grid.x_of_node[channel_nodes]
    channel_y = grid.y_of_node[channel_nodes]
    
    # Define outlet elevation
    if outlet_method == 'min_elevation':
        outlet_elevation = np.min(channel_elevations)
    elif outlet_method == 'percentile':
        outlet_elevation = np.percentile(channel_elevations, 5)
    else:
        outlet_elevation = outlet_method  # Assume it's a number
    
    # Interpolate from channels to create reference surface
    # Use channel elevations, but ensure they don't go below outlet
    ref_elevations = np.maximum(channel_elevations, outlet_elevation)
    
    # Interpolate to all grid nodes
    z_ref = griddata(
        (channel_x, channel_y), ref_elevations,
        (grid.x_of_node, grid.y_of_node),
        method='linear', fill_value=outlet_elevation
    )
    
    # Calculate excess
    z = grid.at_node['topographic__elevation']
    excess = z - z_ref
    excess[excess < 0] = 0
    
    return excess

def _valley_based_reference(grid, percentile=10):
    """
    Calculate excess topography using valley bottoms as reference.
    Assumes flow routing already done with PriorityFloodFlowRouter.
    """
    # Check if we have existing flow fields to help identify valleys
    if 'drainage_area' in grid.at_node:
        drainage_area = grid.at_node['drainage_area']
        if np.max(drainage_area) < 1:  # Convert to actual area if needed
            drainage_area = drainage_area * grid.dx * grid.dy
    else:
        drainage_area = None
    
    # Get topographic position index or use low elevations
    z = grid.at_node['topographic__elevation']
    
    # Enhanced valley detection using both elevation and drainage area
    if drainage_area is not None:
        # Combine low elevation with high drainage area for better valley detection
        elevation_percentile = np.percentile(z[~np.isnan(z)], percentile)
        drainage_percentile = np.percentile(drainage_area[drainage_area > 0], 90)
        
        # Valleys are areas with either very low elevation OR high drainage area
        valley_mask = (z <= elevation_percentile) | (drainage_area >= drainage_percentile)
    else:
        # Fallback to simple elevation-based approach
        valley_threshold = np.percentile(z[~np.isnan(z)], percentile)
        valley_mask = z <= valley_threshold
    
    valley_nodes = np.where(valley_mask)[0]
    
    if len(valley_nodes) == 0:
        raise ValueError("No valley nodes found")
    
    # Get valley coordinates and elevations
    valley_x = grid.x_of_node[valley_nodes]
    valley_y = grid.y_of_node[valley_nodes]
    valley_z = z[valley_nodes]
    
    # Interpolate from valleys to create reference surface
    z_ref = griddata(
        (valley_x, valley_y), valley_z,
        (grid.x_of_node, grid.y_of_node),
        method='linear', fill_value=np.min(valley_z)
    )
    
    # Calculate excess
    excess = z - z_ref
    excess[excess < 0] = 0
    
    return excess

def _morphological_reference(grid, kernel_size=5, method='opening', 
                            gradient_constraint=True, max_gradient=0.3, 
                            iterations=1, fill_method='nearest'):
    """
    Calculate excess topography using morphological operations (TopoToolbox-style).
    
    Parameters:
    -----------
    grid : RasterModelGrid
        Landlab grid with topographic__elevation field
    kernel_size : int
        Size of the morphological structuring element (neighborhood)
    method : str
        'erosion' (default), 'dilation', or 'opening' (erosion followed by dilation)
    gradient_constraint : bool
        Whether to apply gradient constraints during morphological operations
    max_gradient : float
        Maximum allowed gradient (slope) for the reference surface
    iterations : int
        Number of iterations for morphological operations
    fill_method : str
        Method for filling NaN values ('nearest', 'linear', or 'cubic')
    
    Returns:
    --------
    excess : numpy array
        Excess topography values at each node
    """
    
    # Get elevation data and reshape to 2D grid
    z = grid.at_node['topographic__elevation']
    z_2d = z.reshape(grid.shape)
    
    # Handle NaN values
    nan_mask = np.isnan(z_2d)
    z_filled = z_2d.copy()
    
    if np.any(nan_mask):
        if fill_method == 'nearest':
            # Simple nearest neighbor filling
            valid_points = np.where(~nan_mask)
            nan_points = np.where(nan_mask)
            
            if len(valid_points[0]) > 0:
                from scipy.spatial import cKDTree
                tree = cKDTree(np.column_stack(valid_points))
                _, nearest_idx = tree.query(np.column_stack(nan_points))
                z_filled[nan_mask] = z_2d[valid_points][nearest_idx]
        else:
            # Use scipy's griddata for linear/cubic interpolation
            from scipy.interpolate import griddata
            x_2d, y_2d = grid.xy_of_node.reshape((2,) + grid.shape)
            valid_mask = ~nan_mask
            points = np.column_stack([x_2d[valid_mask], y_2d[valid_mask]])
            values = z_2d[valid_mask]
            xi = np.column_stack([x_2d[nan_mask], y_2d[nan_mask]])
            z_filled[nan_mask] = griddata(points, values, xi, method=fill_method)
    
    # Create structuring element (kernel)
    kernel = _create_disk_kernel(kernel_size)
    
    # Apply morphological operations
    z_ref = z_filled.copy()
    
    for i in range(iterations):
        if method == 'erosion':
            z_ref = grey_erosion(z_ref, structure=kernel)
        elif method == 'dilation':
            z_ref = grey_dilation(z_ref, structure=kernel)
        elif method == 'opening':
            z_ref = grey_erosion(z_ref, structure=kernel)
            z_ref = grey_dilation(z_ref, structure=kernel)
        
        # Apply gradient constraint
        if gradient_constraint:
            z_ref = _apply_gradient_constraint(z_ref, grid.dx, grid.dy, max_gradient)
    
    # Flatten back to 1D array
    z_ref_1d = z_ref.flatten()
    
    # Calculate excess topography
    excess = z - z_ref_1d
    excess[excess < 0] = 0  # Only positive excess
    
    # Set NaN where original data was NaN
    excess[np.isnan(z)] = np.nan
    
    return excess

def _create_disk_kernel(size):
    """Create a disk-shaped structuring element."""
    y, x = np.ogrid[-size:size+1, -size:size+1]
    kernel = x**2 + y**2 <= size**2
    return kernel.astype(np.uint8)

def _apply_gradient_constraint(z, dx, dy, max_gradient):
    """
    Apply gradient constraint to ensure reference surface doesn't exceed
    maximum gradient (similar to TopoToolbox's approach).
    """
    # Calculate gradients
    grad_y, grad_x = np.gradient(z, dy, dx)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Find areas where gradient exceeds maximum
    steep_mask = gradient_magnitude > max_gradient
    
    if np.any(steep_mask):
        # Apply smoothing to areas with excessive gradients
        # Use a simple approach: weighted average with neighbors
        z_smooth = gaussian_filter(z, sigma=1)
        
        # Blend original and smoothed where gradients are too steep
        blend_factor = np.minimum(1.0, gradient_magnitude / max_gradient)
        z_constrained = z * (1 - blend_factor) + z_smooth * blend_factor
        
        return z_constrained
    
    return z