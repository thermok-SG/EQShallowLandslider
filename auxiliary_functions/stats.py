"""
Functions for statistical analysis of landslide areas
"""

# %% Required components
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import pandas as pd
from skimage.measure import regionprops

# %% Split regions by width
def split_wide_regions(labeled_array, region_df, kde_results, transform_info, width_threshold=1.5,
                        label_col='label', length_col='length_m', width_col='width_m'):
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

# %% 
def calculate_region_dimensions(labeled_array, elevation_grid, aspect_array, slopes_grid, grid, unique_labels=None):
    """
    Calculate length and width dimensions for labeled regions using elevation-based methods.
    Fixed version that properly handles coordinate systems and slope directions.
    
    Parameters:
    -----------
    labeled_array : numpy.ndarray
        2D array of integer labels where each region has a unique label
    elevation_grid : numpy.ndarray
        2D array of elevation values
    aspect_array : numpy.ndarray
        2D array of aspect values in degrees (geographic convention: 0=North, 90=East)
    slopes_grid : numpy.ndarray
        2D array of slope values
    grid : object
        Grid object with dx, dy attributes for coordinate conversion
    unique_labels : array-like, optional
        Labels to process. If None, all unique labels except 0 are used.
        
    Returns:
    --------
    dict
        Dictionary containing dimension measurements for each region
    """
    elevation_grid = elevation_grid.reshape(grid.shape)
    slopes_grid = slopes_grid.reshape(grid.shape)
    aspect_array = aspect_array.reshape(grid.shape)
    
    if unique_labels is None:
        unique_labels = np.unique(labeled_array)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background
    
    # Get basic region properties using regionprops
    regions = regionprops(labeled_array)
    
    # Extract properties manually
    labels = []
    areas = []
    min_elevations = []
    max_elevations = []
    
    for region in regions:
        labels.append(region.label)
        areas.append(region.area)
        
        # Get elevation values for this region
        region_elevations = elevation_grid[region.coords[:, 0], region.coords[:, 1]]
        min_elevations.append(np.min(region_elevations))
        max_elevations.append(np.max(region_elevations))
    
    # Convert to numpy arrays
    labels = np.array(labels)
    areas = np.array(areas)
    min_elevations = np.array(min_elevations)
    max_elevations = np.array(max_elevations)
    
    # Convert to dict format for easier manipulation
    result_props = {
        'label': labels,
        'area': areas,
        'min_elevation': min_elevations,
        'max_elevation': max_elevations,
        'local_relief': max_elevations - min_elevations,
        'slope_direction_length_new': np.zeros(len(labels)),
        'perpendicular_width_new': np.zeros(len(labels)),
        'direction_method': [''] * len(labels)
    }
    
    relief_threshold = 2.0  # meters - adjust based on your data
    
    for i, label_num in enumerate(unique_labels):
        if label_num == 0:  # Skip background
            continue
            
        # Find the index in our props arrays
        prop_idx = np.where(result_props['label'] == label_num)[0]
        if len(prop_idx) == 0:
            continue
        prop_idx = prop_idx[0]
        
        elevation_relief = result_props['local_relief'][prop_idx]
        
        region_mask = labeled_array == label_num
        row_coords, col_coords = np.where(region_mask)
        
        if len(row_coords) == 0:
            continue
        
        # FIXED: Proper coordinate conversion
        # In numpy arrays: rows = y-direction, cols = x-direction
        # Convert to real-world coordinates (assuming y increases northward, x increases eastward)
        x_coords = col_coords * grid.dx  # East-West direction
        y_coords = row_coords * grid.dy  # North-South direction
        coords = np.column_stack([x_coords, y_coords])  # [x, y] pairs
        
        gradient_direction = None
        method_used = None
        
        # Method 1: Use elevation gradient if there's significant relief
        if elevation_relief > relief_threshold:
            region_elevations = elevation_grid[region_mask]
            
            max_elevation = result_props['max_elevation'][prop_idx]
            min_elevation = result_props['min_elevation'][prop_idx]
            elevation_tolerance = 0.1 * elevation_relief  # Scale tolerance with relief
            
            # Find points at max and min elevations
            high_indices = region_elevations >= (max_elevation - elevation_tolerance)
            low_indices = region_elevations <= (min_elevation + elevation_tolerance)
            
            high_coords = coords[high_indices]
            low_coords = coords[low_indices]
            
            if len(high_coords) > 0 and len(low_coords) > 0:
                high_centroid = np.mean(high_coords, axis=0)
                low_centroid = np.mean(low_coords, axis=0)
                
                # FIXED: Calculate gradient vector (points downslope)
                downslope_vector = low_centroid - high_centroid
                gradient_length = np.linalg.norm(downslope_vector)
                
                if gradient_length > 0:
                    # Use the downslope direction as the primary direction
                    # This assumes regions are elongated along the slope direction
                    gradient_direction = downslope_vector / gradient_length
                    method_used = 'elevation_gradient'
        
        # Method 2: Fallback to aspect-based calculation
        if gradient_direction is None:
            region_aspects = aspect_array[region_mask]
            region_slopes = slopes_grid[region_mask]
            
            # Filter out low-slope areas for more reliable aspect calculation
            slope_threshold = max(np.percentile(region_slopes, 25), 1.0)  # At least 1 degree
            valid_mask = region_slopes > slope_threshold
            
            if np.sum(valid_mask) > 0:
                valid_aspects = region_aspects[valid_mask]
                valid_slopes = region_slopes[valid_mask]
                
                # FIXED: Proper aspect to direction conversion
                # Geographic aspect: 0° = North, 90° = East, 180° = South, 270° = West
                # Convert to mathematical convention for consistent coordinate system
                # Aspect is the direction the slope faces (upslope direction)
                # We want the downslope direction (opposite of aspect)
                
                # Convert aspect to downslope direction in radians
                downslope_aspects_rad = (valid_aspects + 180) * np.pi / 180  # Add 180° for downslope
                
                # Calculate slope-weighted mean direction
                cos_aspects = np.cos(downslope_aspects_rad) * valid_slopes
                sin_aspects = np.sin(downslope_aspects_rad) * valid_slopes
                
                mean_cos = np.sum(cos_aspects) / np.sum(valid_slopes)
                mean_sin = np.sum(sin_aspects) / np.sum(valid_slopes)
                
                # FIXED: Convert from geographic to cartesian coordinates
                # Geographic: 0° = North (positive y), 90° = East (positive x)
                # Cartesian: 0° = East (positive x), 90° = North (positive y)
                mean_direction_rad = np.arctan2(mean_sin, mean_cos)
                
                # Convert from geographic convention to cartesian (x=East, y=North)
                # Geographic angle to cartesian: subtract from π/2
                cartesian_angle = np.pi/2 - mean_direction_rad
                
                gradient_direction = np.array([np.cos(cartesian_angle), np.sin(cartesian_angle)])
                method_used = 'aspect_weighted'
            else:
                # Final fallback to simple aspect mean
                # Remove any NaN values
                valid_aspects = region_aspects[~np.isnan(region_aspects)]
                if len(valid_aspects) > 0:
                    # Convert to downslope direction
                    downslope_aspects_rad = (valid_aspects + 180) * np.pi / 180
                    
                    cos_aspects = np.cos(downslope_aspects_rad)
                    sin_aspects = np.sin(downslope_aspects_rad)
                    mean_direction_rad = np.arctan2(np.mean(sin_aspects), np.mean(cos_aspects))
                    
                    # Convert to cartesian
                    cartesian_angle = np.pi/2 - mean_direction_rad
                    gradient_direction = np.array([np.cos(cartesian_angle), np.sin(cartesian_angle)])
                    method_used = 'aspect_simple'
                else:
                    # Ultimate fallback: use the major axis of the region
                    region_props = regionprops(labeled_array == label_num)[0]
                    orientation = region_props.orientation  # Angle in radians
                    gradient_direction = np.array([np.cos(orientation), np.sin(orientation)])
                    method_used = 'region_orientation'
        
        # If we still don't have a direction, use region orientation
        if gradient_direction is None:
            region_props = regionprops(labeled_array == label_num)[0]
            orientation = region_props.orientation
            gradient_direction = np.array([np.cos(orientation), np.sin(orientation)])
            method_used = 'region_orientation_fallback'
        
        # Calculate projections
        perp_direction = np.array([-gradient_direction[1], gradient_direction[0]])
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        
        gradient_projections = np.dot(centered_coords, gradient_direction)
        perp_projections = np.dot(centered_coords, perp_direction)
        
        result_props['slope_direction_length_new'][prop_idx] = np.max(gradient_projections) - np.min(gradient_projections)
        result_props['perpendicular_width_new'][prop_idx] = np.max(perp_projections) - np.min(perp_projections)
        result_props['direction_method'][prop_idx] = method_used
    
    return result_props

def split_wide_regions_single_iteration(labeled_array, region_df, kde_results, transform_info, 
                                        width_threshold=1.5, label_col='label', 
                                        length_col='length_m', width_col='width_m'):
    """
    Perform a single iteration of region splitting.
    Modified version of the original split_wide_regions function.
    """
    region_df = region_df.reset_index(drop=True)
    
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
    
    # Identify which regions need splitting
    regions_to_split = []
    
    for _, row in region_df.iterrows():
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
        num_samples = 200
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
    
    # Split the identified regions
    for region in regions_to_split:
        label_id = region['label']
        
        # Get region mask
        mask = labeled_array == label_id
        
        # Find coordinates and centroid
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            continue  # Skip empty regions
            
        centroid_x = np.mean(x_coords)
        
        # Create the split using array indices
        rows, cols = np.indices(labeled_array.shape)
        
        # Create left and right masks
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
    
    return new_labels, split_info

def recursive_split_wide_regions(grid, labeled_array, aspect_array, slopes_grid, 
                                kde_results, transform_info, width_threshold=1.5,
                                max_iterations=10, min_region_size=10, 
                                convergence_threshold=0.95, verbose=True):
    """
    Recursively split labeled regions where the actual width is significantly larger than
    the width expected from the KDE distribution, recalculating dimensions after each split.
    
    Parameters:
    -----------
    labeled_array : numpy.ndarray
        2D array of integer labels where each region has a unique label
    elevation_grid : numpy.ndarray
        2D array of elevation values
    aspect_array : numpy.ndarray
        2D array of aspect values in degrees
    slopes_grid : numpy.ndarray
        2D array of slope values
    grid : object
        Grid object with dx, dy attributes for coordinate conversion
    kde_results : dict
        Dictionary with KDE objects from fit_bivariate_kde function
    transform_info : dict
        Information about transformations used in the KDE
    width_threshold : float, optional
        Ratio of actual width to KDE-expected width above which regions are split (default: 1.5)
    max_iterations : int, optional
        Maximum number of splitting iterations (default: 10)
    min_region_size : int, optional
        Minimum number of pixels a region must have to be considered for splitting (default: 10)
    convergence_threshold : float, optional
        Fraction of regions that must conform to stop iteration (default: 0.95)
    verbose : bool, optional
        Whether to print progress information (default: True)
        
    Returns:
    --------
    numpy.ndarray
        Final labeled array after all splits (matches split_wide_regions output format)
    list
        Complete information about all split operations (matches split_wide_regions output format)
    """
    
    elevation_grid = grid.at_node["topographic__elevation"]
    current_labels = labeled_array.copy()
    all_split_info = []
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n=== Iteration {iteration + 1} ===")
        
        # Calculate dimensions for current regions
        unique_labels = np.unique(current_labels)
        unique_labels = unique_labels[unique_labels != 0]  # Remove background
        
        if verbose:
            print(f"Calculating dimensions for {len(unique_labels)} regions...")
        
        props = calculate_region_dimensions(current_labels, elevation_grid, aspect_array, 
                                            slopes_grid, grid, unique_labels)
        
        # Create DataFrame for easier handling
        region_df = pd.DataFrame({
            'label': props['label'],
            'length_m': props['slope_direction_length_new'],
            'width_m': props['perpendicular_width_new'],
            'area': props['area'],
            'direction_method': props['direction_method']
        })
        
        # Filter out regions that are too small
        region_df = region_df[region_df['area'] >= min_region_size]
        
        if len(region_df) == 0:
            if verbose:
                print("No regions left to process.")
            break
        
        # Perform one iteration of splitting
        new_labels, split_info = split_wide_regions_single_iteration(
            current_labels, region_df, kde_results, transform_info, width_threshold
        )
        
        # Check for convergence
        num_splits = len(split_info)
        total_regions = len(region_df)
        conforming_regions = total_regions - num_splits
        conformance_rate = conforming_regions / total_regions if total_regions > 0 else 1.0
        
        if verbose:
            print(f"Split {num_splits} regions out of {total_regions} total regions")
            print(f"Conformance rate: {conformance_rate:.1%}")
        
        # Store split information with iteration number added
        for split in split_info:
            split['iteration'] = iteration + 1
        all_split_info.extend(split_info)
        
        # Check convergence
        if num_splits == 0:
            if verbose:
                print("No more regions need splitting. Converged!")
            break
        elif conformance_rate >= convergence_threshold:
            if verbose:
                print(f"Reached convergence threshold ({convergence_threshold:.1%}). Stopping.")
            break
        
        # Update for next iteration
        current_labels = new_labels
    
    if verbose:
        print("\nCompleted recursive splitting:")
        print(f"Total iterations: {len(set(split['iteration'] for split in all_split_info)) if all_split_info else 0}")
        print(f"Total splits performed: {len(all_split_info)}")
        
        # Count final regions
        final_unique_labels = np.unique(current_labels)
        final_unique_labels = final_unique_labels[final_unique_labels != 0]
        print(f"Final number of regions: {len(final_unique_labels)}")
    
    return current_labels, all_split_info

def analyze_split_results(split_history, final_df, kde_results, transform_info):
    """
    Analyze the results of the recursive splitting process.
    
    Parameters:
    -----------
    split_history : list
        Complete history of all split operations
    final_df : pandas.DataFrame
        Final region measurements
    kde_results : dict
        KDE results for comparison
    transform_info : dict
        Transform information from KDE
        
    Returns:
    --------
    dict
        Analysis results including statistics and conformance rates
    """
    
    # Calculate final conformance
    kde = kde_results['overall']
    log_x = transform_info.get('log_x', False)
    log_y = transform_info.get('log_y', False)
    
    conforming_regions = 0
    width_ratios = []
    
    for _, row in final_df.iterrows():
        length = row['length_m']
        actual_width = row['width_m']
        
        if length <= 0 or actual_width <= 0:
            continue
            
        # Get expected width from KDE
        if log_x:
            length_t = np.log(length)
        else:
            length_t = length
            
        # Sample from KDE
        samples = kde.resample(100)[1, :]
        
        if log_y:
            expected_widths = np.exp(samples)
        else:
            expected_widths = samples
            
        expected_width = np.median(expected_widths)
        width_ratio = actual_width / expected_width
        width_ratios.append(width_ratio)
        
        if width_ratio <= 1.5:  # Using same threshold as splitting
            conforming_regions += 1
    
    total_regions = len(final_df)
    conformance_rate = conforming_regions / total_regions if total_regions > 0 else 0
    
    # Split statistics by iteration
    iterations = {}
    for split in split_history:
        iter_num = split['iteration']
        if iter_num not in iterations:
            iterations[iter_num] = []
        iterations[iter_num].append(split)
    
    results = {
        'total_splits': len(split_history),
        'total_iterations': len(iterations),
        'final_regions': total_regions,
        'final_conformance_rate': conformance_rate,
        'width_ratios': width_ratios,
        'splits_by_iteration': {k: len(v) for k, v in iterations.items()},
        'mean_width_ratio': np.mean(width_ratios) if width_ratios else 0,
        'max_width_ratio': np.max(width_ratios) if width_ratios else 0
    }
    
    return results

# %% Statistical functions

# %%% Bivariate kde
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
    
    plt.title("Overall Bivariate KDE in Transformed Space")
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
    plt.axline([1,1],[10,10], label='1:1', linestyle='--', color='black')
    
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

# %%% Sample from kde
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

# %%%% Plot results of conditional sample from kde
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

# %%%% Plot all conditional samples
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
                