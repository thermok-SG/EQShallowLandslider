"""
Region splitting functions based on measured KDEs

Author: sghoshal
"""
# %% Import required packages

import numpy as np
import pandas as pd
from skimage.measure import regionprops


# %% Splitting regions based on length/width ratio
# %%% Splitting - main function
def recursive_split_wide_regions(
    grid,
    labeled_array,
    aspect_array,
    slopes_grid,
    kde_results,
    transform_info,
    width_threshold=1.5,
    max_iterations=10,
    min_region_size=10,
    convergence_threshold=0.95,
    verbose: bool = False,
):
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

        props = calculate_region_dimensions(
            current_labels,
            elevation_grid,
            aspect_array,
            slopes_grid,
            grid,
            unique_labels,
        )

        # Create DataFrame for easier handling
        region_df = pd.DataFrame(
            {
                "label": props["label"],
                "length_m": props["slope_direction_length_new"],
                "width_m": props["perpendicular_width_new"],
                "area": props["area"],
                "direction_method": props["direction_method"],
            }
        )

        # Filter out regions that are too small
        region_df = region_df[region_df["area"] >= min_region_size]

        if len(region_df) == 0:
            if verbose:
                print("No regions left to process.")
            break

        # Perform one iteration of splitting
        new_labels, split_info = split_wide_regions_single_iteration(
            grid,
            current_labels,
            region_df,
            kde_results,
            transform_info,
            width_threshold,
        )

        # Check for convergence
        num_splits = len(split_info)
        total_regions = len(region_df)
        conforming_regions = total_regions - num_splits
        conformance_rate = (
            conforming_regions / total_regions if total_regions > 0 else 1.0
        )

        if verbose:
            print(f"Split {num_splits} regions out of {total_regions} total regions")
            print(f"Conformance rate: {conformance_rate:.1%}")

        # Store split information with iteration number added
        for split in split_info:
            split["iteration"] = iteration + 1
        all_split_info.extend(split_info)

        # Check convergence
        if num_splits == 0:
            if verbose:
                print("No more regions need splitting. Converged!")
            break
        elif conformance_rate >= convergence_threshold:
            if verbose:
                print(
                    f"Reached convergence threshold ({convergence_threshold:.1%}). Stopping."
                )
            break

        # Update for next iteration
        current_labels = new_labels

    if verbose:
        print("\nCompleted recursive splitting:")
        print(
            f"Total iterations: {len(set(split['iteration'] for split in all_split_info)) if all_split_info else 0}"
        )
        print(f"Total splits performed: {len(all_split_info)}")

        # Count final regions
        final_unique_labels = np.unique(current_labels)
        final_unique_labels = final_unique_labels[final_unique_labels != 0]
        print(f"Final number of regions: {len(final_unique_labels)}")

    return current_labels, all_split_info


# %% Region dimension calculation
def calculate_region_dimensions(
    labeled_array, elevation_grid, aspect_array, slopes_grid, grid, unique_labels=None
):
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
        "label": labels,
        "area": areas,
        "min_elevation": min_elevations,
        "max_elevation": max_elevations,
        "local_relief": max_elevations - min_elevations,
        "slope_direction_length_new": np.zeros(len(labels)),
        "perpendicular_width_new": np.zeros(len(labels)),
        "direction_method": [""] * len(labels),
    }

    relief_threshold = 2.0  # meters - adjust based on your data

    for i, label_num in enumerate(unique_labels):
        if label_num == 0:  # Skip background
            continue

        # Find the index in our props arrays
        prop_idx = np.where(result_props["label"] == label_num)[0]
        if len(prop_idx) == 0:
            continue
        prop_idx = prop_idx[0]

        elevation_relief = result_props["local_relief"][prop_idx]

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

            max_elevation = result_props["max_elevation"][prop_idx]
            min_elevation = result_props["min_elevation"][prop_idx]
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
                    method_used = "elevation_gradient"

        # Method 2: Fallback to aspect-based calculation
        if gradient_direction is None:
            region_aspects = aspect_array[region_mask]
            region_slopes = slopes_grid[region_mask]

            # Filter out low-slope areas for more reliable aspect calculation
            slope_threshold = max(
                np.percentile(region_slopes, 25), 1.0
            )  # At least 1 degree
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
                downslope_aspects_rad = (
                    (valid_aspects + 180) * np.pi / 180
                )  # Add 180° for downslope

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
                cartesian_angle = np.pi / 2 - mean_direction_rad

                gradient_direction = np.array(
                    [np.cos(cartesian_angle), np.sin(cartesian_angle)]
                )
                method_used = "aspect_weighted"
            else:
                # Final fallback to simple aspect mean
                # Remove any NaN values
                valid_aspects = region_aspects[~np.isnan(region_aspects)]
                if len(valid_aspects) > 0:
                    # Convert to downslope direction
                    downslope_aspects_rad = (valid_aspects + 180) * np.pi / 180

                    cos_aspects = np.cos(downslope_aspects_rad)
                    sin_aspects = np.sin(downslope_aspects_rad)
                    mean_direction_rad = np.arctan2(
                        np.mean(sin_aspects), np.mean(cos_aspects)
                    )

                    # Convert to cartesian
                    cartesian_angle = np.pi / 2 - mean_direction_rad
                    gradient_direction = np.array(
                        [np.cos(cartesian_angle), np.sin(cartesian_angle)]
                    )
                    method_used = "aspect_simple"
                else:
                    # Ultimate fallback: use the major axis of the region
                    region_props = regionprops(labeled_array == label_num)[0]
                    orientation = region_props.orientation  # Angle in radians
                    gradient_direction = np.array(
                        [np.cos(orientation), np.sin(orientation)]
                    )
                    method_used = "region_orientation"

        # If we still don't have a direction, use region orientation
        if gradient_direction is None:
            region_props = regionprops(labeled_array == label_num)[0]
            orientation = region_props.orientation
            gradient_direction = np.array([np.cos(orientation), np.sin(orientation)])
            method_used = "region_orientation_fallback"

        # Calculate projections
        perp_direction = np.array([-gradient_direction[1], gradient_direction[0]])
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid

        gradient_projections = np.dot(centered_coords, gradient_direction)
        perp_projections = np.dot(centered_coords, perp_direction)

        result_props["slope_direction_length_new"][prop_idx] = np.max(
            gradient_projections
        ) - np.min(gradient_projections)
        result_props["perpendicular_width_new"][prop_idx] = np.max(
            perp_projections
        ) - np.min(perp_projections)
        result_props["direction_method"][prop_idx] = method_used

    return result_props


# %%% Splitting each region
def split_wide_regions_single_iteration(
    grid,
    labeled_array,
    region_df,
    kde_results,
    transform_info,
    width_threshold=1.5,
    label_col="label",
    length_col="length_m",
    width_col="width_m",
):
    """
    Perform a single iteration of region splitting.
    Modified to split perpendicular to the region's main axis using pre-calculated directions.

    The region_df should include direction information from calculate_region_dimensions.
    """

    # Verify the column names exist in the DataFrame
    required_cols = [label_col, length_col, width_col]
    for col_name, col_desc in zip(required_cols, ["label", "length", "width"]):
        if col_name not in region_df.columns:
            raise ValueError(
                f"Column '{col_name}' specified for {col_desc} not found in the DataFrame. "
                f"Available columns are: {list(region_df.columns)}"
            )

    # Start with a copy of the original labeled array
    new_labels = labeled_array.copy()
    next_label = np.max(labeled_array) + 1 if labeled_array.size > 0 else 1
    split_info = []

    # Get KDE information
    kde = kde_results["overall"]
    log_x = transform_info.get("log_x", False)
    log_y = transform_info.get("log_y", False)

    # Identify which regions need splitting and get their directions
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
            # Extract the main axis direction from calculate_region_dimensions
            main_direction = extract_direction_from_region_data(
                row, labeled_array, grid
            )

            regions_to_split.append(
                {
                    "label": label_id,
                    "actual_width": actual_width,
                    "expected_width": expected_width,
                    "ratio": width_ratio,
                    "main_direction": main_direction,
                    "direction_method": row.get("direction_method", "unknown"),
                }
            )

    # Split the identified regions
    for region in regions_to_split:
        label_id = region["label"]
        main_direction = region["main_direction"]

        # Get region mask
        mask = labeled_array == label_id

        # Find coordinates
        row_coords, col_coords = np.where(mask)
        if len(row_coords) == 0:
            continue  # Skip empty regions

        # Convert to real-world coordinates
        x_coords = col_coords * grid.dx  # East-West direction
        y_coords = row_coords * grid.dy  # North-South direction
        coords = np.column_stack([x_coords, y_coords])  # [x, y] pairs

        # The split should be perpendicular to the main axis direction
        if main_direction is not None:
            split_direction = np.array([-main_direction[1], main_direction[0]])
        else:
            # Fallback to PCA if no direction available
            split_direction = get_pca_split_direction(coords)

        # Calculate centroid
        centroid = np.mean(coords, axis=0)

        # Project all points onto the split direction to determine which side they're on
        centered_coords = coords - centroid
        split_projections = np.dot(centered_coords, split_direction)

        # Create masks for the two parts
        split_mask = split_projections >= 0

        # Convert back to 2D indices
        part1_rows = row_coords[split_mask]
        part1_cols = col_coords[split_mask]
        part2_rows = row_coords[~split_mask]
        part2_cols = col_coords[~split_mask]

        # Ensure both parts have pixels
        if len(part1_rows) == 0 or len(part2_rows) == 0:
            continue  # Skip if one half would be empty

        # Create masks for the new labels
        part1_mask = np.zeros_like(mask, dtype=bool)
        part2_mask = np.zeros_like(mask, dtype=bool)

        part1_mask[part1_rows, part1_cols] = True
        part2_mask[part2_rows, part2_cols] = True

        # Assign new labels
        new_labels[part1_mask] = next_label
        part1_label = next_label
        next_label += 1

        new_labels[part2_mask] = next_label
        part2_label = next_label
        next_label += 1

        # Record the split
        region["part1_label"] = part1_label
        region["part2_label"] = part2_label
        region["original_size"] = len(row_coords)
        region["part1_size"] = len(part1_rows)
        region["part2_size"] = len(part2_rows)
        region["split_direction"] = (
            split_direction.tolist() if main_direction is not None else None
        )
        split_info.append(region)

    return new_labels, split_info


# %%%% Splitting - helper functions
def extract_direction_from_region_data(region_row, labeled_array, grid):
    """
    Extract the main axis direction from the region data calculated by calculate_region_dimensions.

    This function reconstructs the direction vector from the region's geometry,
    using the same coordinate system as calculate_region_dimensions.
    """
    label_id = region_row["label"]
    direction_method = region_row.get("direction_method", "")

    # Get region coordinates
    mask = labeled_array == label_id
    row_coords, col_coords = np.where(mask)

    if len(row_coords) == 0:
        return None

    # Convert to real-world coordinates (same as calculate_region_dimensions)
    x_coords = col_coords * grid.dx  # East-West direction
    y_coords = row_coords * grid.dy  # North-South direction
    coords = np.column_stack([x_coords, y_coords])

    # For most direction methods, we can reconstruct the direction from the region's shape
    # This is more reliable than trying to store/pass the direction vector

    if "elevation_gradient" in direction_method:
        # The direction was based on elevation gradient - use region's major axis
        # as it should align with the gradient direction
        return _get_region_major_axis(coords)

    elif "aspect" in direction_method:
        # Direction was based on aspect - use region's major axis
        return _get_region_major_axis(coords)

    elif "orientation" in direction_method:
        # Direction was based on region orientation - use major axis directly
        return _get_region_major_axis(coords)

    else:
        # Unknown method - use major axis as fallback
        return _get_region_major_axis(coords)


def _get_region_major_axis(coords):
    """
    Calculate the major axis direction of a set of coordinates using PCA.
    Returns a unit vector pointing along the major axis.
    """
    if len(coords) < 2:
        return np.array([1, 0])  # Default to east-west

    # Center the coordinates
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # Calculate covariance matrix
    cov_matrix = np.cov(centered_coords.T)

    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # The eigenvector with the largest eigenvalue is the major axis
    major_axis_idx = np.argmax(eigenvalues)
    major_axis = eigenvectors[:, major_axis_idx]

    # Ensure consistent orientation (optional - you might want to remove this)
    if major_axis[0] < 0:
        major_axis = -major_axis

    return major_axis


def get_pca_split_direction(coords):
    """
    Fallback function to get split direction when no main direction is available.
    Returns the direction perpendicular to the major axis.
    """
    major_axis = _get_region_major_axis(coords)
    # Return perpendicular direction
    return np.array([-major_axis[1], major_axis[0]])
