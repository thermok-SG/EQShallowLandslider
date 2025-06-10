"""
Functions for discrete region identification and analysis 
"""
# %% Required packages
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.ndimage import (
    label, gaussian_filter, binary_dilation,
    generate_binary_structure, center_of_mass,
    labeled_comprehension
    )
from skimage.measure import regionprops


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
    
    print("Splitting groups by aspect")
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

# %% Calculate properties for selected regions
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