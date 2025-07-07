"""
Functions for calculating Excess topography from landlab grid

"""
import numpy as np
from scipy.interpolate import griddata

from scipy.ndimage import (
    grey_erosion, grey_dilation,
    gaussian_filter
)
from scipy.spatial.distance import cdist

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