"""
Landslide selection functions for ShallowLandslider

Author: sghoshal
"""

# %% Import required packages

import numpy as np
from scipy import ndimage
from scipy.special import expit

# %% "probabilistic" path


def generate_landslide_probability(
    grid,
    h_pga_array,
    v_pga_array,
    labeled_array,
    slope_array=None,
    soil_array=None,
    geological_factor_array=None,
    critical_acceleration_array=None,
    default_critical_acceleration=0.2,
    random_seed=None,
    normalise_final_probs=False,
):
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
            h_pga_grid, default_critical_acceleration, dtype=np.float32
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
            h_pga_grid,
            v_pga_grid,
            critical_acceleration_grid,
            mask,
            slope_grid,
            soil_array,
            geological_factor_array,
        )

        # Store mask for later use
        group_info["mask"] = mask
        group_probs[label_num] = group_info

    # Normalize if requested
    if normalise_final_probs:
        normalized_group_probs, norm_metadata = normalize_group_probabilities(
            group_probs
        )

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
            group_probs, probability_array, {"performed": False}, normalized=False
        )

    return probability_array, metadata


# %%% Group probability calculation
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
def calculate_group_probability(
    h_pga_grid,
    v_pga_grid,
    critical_acceleration_grid,
    mask,
    slope_grid=None,
    soil_array=None,
    geological_factor_array=None,
):
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
    resultant_pga = np.sqrt(np.mean(group_h_pga) ** 2 + np.mean(group_v_pga) ** 2)

    # Acceleration Ratio Calculation
    pga_ratio = resultant_pga / local_critical_acceleration

    # Base Probability Model with Critical Acceleration
    base_prob = _calculate_acceleration_probability(
        pga_ratio, local_critical_acceleration
    )

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
        base_prob *= 1 + 0.5 * geo_factor

    # Stochastic Variability
    stochastic_factor = np.random.lognormal(mean=0, sigma=0.2)
    group_prob = np.clip(base_prob * stochastic_factor, 0, 1)

    return {
        "probability": group_prob,
        "critical_acceleration": local_critical_acceleration,
        "resultant_pga": resultant_pga,
        "pga_ratio": pga_ratio,
        "base_probability": base_prob,
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
    probs = [info["probability"] for info in group_probs.values()]

    if len(probs) <= 1:
        # Nothing to normalize with only one group
        norm_metadata = {"performed": False, "reason": "Only one group present"}
        return group_probs, norm_metadata

    min_prob = min(probs)
    max_prob = max(probs)

    if max_prob <= min_prob:
        # No range to normalize
        norm_metadata = {
            "performed": False,
            "reason": "All groups have the same probability",
            "value": min_prob,
        }
        return group_probs, norm_metadata

    # Normalize each group's probability
    normalized_group_probs = {}
    for label_num, info in group_probs.items():
        # Copy the original info
        normalized_info = info.copy()

        # Apply min-max normalization
        normalized_prob = (info["probability"] - min_prob) / (max_prob - min_prob)
        normalized_info["normalized_probability"] = normalized_prob

        normalized_group_probs[label_num] = normalized_info

    normalized_metadata = {
        "performed": True,
        "min_raw_prob": min_prob,
        "max_raw_prob": max_prob,
    }

    return normalized_group_probs, normalized_metadata


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
        if normalized and "normalized_probability" in info:
            probability_array[info["mask"]] = info["normalized_probability"]
        else:
            probability_array[info["mask"]] = info["probability"]

    return probability_array


# Create final metadata
def create_metadata(
    group_probs, probability_array, normalization_metadata, normalized=False
):
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
    metadata = {"group_details": [], "normalization": normalization_metadata}

    for label_num, info in group_probs.items():
        group_meta = {
            "label": label_num,
            "critical_acceleration": info["critical_acceleration"],
            "resultant_pga": info["resultant_pga"],
            "pga_ratio": info["pga_ratio"],
            "base_probability": info["base_probability"],
        }

        if normalized and "normalized_probability" in info:
            group_meta["raw_probability"] = info["probability"]
            group_meta["final_probability"] = info["normalized_probability"]
        else:
            group_meta["final_probability"] = info["probability"]

        metadata["group_details"].append(group_meta)

    # Calculate overall statistics
    nonzero_probs = probability_array[probability_array > 0]
    if len(nonzero_probs) > 0:
        metadata["overall_proportion"] = np.mean(nonzero_probs)
        metadata["max_proportion"] = np.max(nonzero_probs)
        metadata["min_proportion"] = np.min(nonzero_probs)
    else:
        metadata["overall_proportion"] = 0
        metadata["max_proportion"] = 0
        metadata["min_proportion"] = 0

    return metadata


# %%%% Helper functions for calculate_group_probability


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
    if critical_acceleration <= 0:
        return 1.0

    else:
        epsilon = 1e-3
        # Base probability increases as critical acceleration decreases
        exponent = -5 * (1.0 / max(critical_acceleration, epsilon))
        exponent = np.clip(exponent, -700, 700)
        base_probability = 1 - np.exp(exponent)

    # PGA ratio effect
    pga_effect = expit(5 * (pga_ratio - 1))

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


# %%% Region selection using calculated probabilities


def probabilistic_group_selection(
    labeled_array,
    probability_array,
    proportion_method="empirical",
    custom_proportion=None,
    random_seed=5000,
    reproducible=True,
    verbose: bool = False,
):
    """
    Enhanced probabilistic group selection with dynamic proportion calculation.
    """
    unique_labels = np.unique(labeled_array)
    unique_labels = unique_labels[unique_labels != 0]
    num_groups = len(unique_labels)

    if reproducible and random_seed is not None:
        np.random.seed(random_seed)

    if verbose:
        print(f"Debug: Total groups available: {num_groups}")

    # Compute group-level mean probabilities efficiently
    group_probs = ndimage.mean(
        probability_array, labels=labeled_array, index=unique_labels
    )
    group_probs = np.array(group_probs)

    # Determine proportion
    if custom_proportion is not None:
        proportion = custom_proportion
        method = "user_defined"
        if verbose:
            print(f"Debug: Using custom proportion: {proportion}")
    else:
        proportion = _calculate_landslide_proportion(
            group_probs, method=proportion_method
        )
        method = proportion_method

    if np.sum(group_probs) == 0:
        if verbose:
            print("Warning: All group probabilities are zero!")
        return np.zeros_like(labeled_array), {
            "method_used": method,
            "proportion_calculated": 0.0,
            "num_groups_total": num_groups,
            "num_groups_selected": 0,
            "selected_labels": [],
            "group_probabilities": {},
            "selection_probabilities": {},
        }

    normalized_probs = group_probs / np.sum(group_probs)
    num_to_select = max(1, int(np.ceil(num_groups * proportion)))
    num_to_select = min(num_to_select, num_groups)

    if verbose:
        print(
            f"Debug: Proportion={proportion:.4f}, Selecting {num_to_select} out of {num_groups} groups"
        )

    selected_labels = np.random.choice(
        unique_labels, num_to_select, replace=False, p=normalized_probs
    )
    selected_groups = np.isin(labeled_array, selected_labels) * labeled_array

    metadata = {
        "method_used": method,
        "proportion_calculated": proportion,
        "num_groups_total": num_groups,
        "num_groups_selected": num_to_select,
        "selected_labels": selected_labels.tolist(),
        "group_probabilities": dict(zip(unique_labels.tolist(), group_probs.tolist())),
        "selection_probabilities": dict(
            zip(unique_labels.tolist(), normalized_probs.tolist())
        ),
    }

    return selected_groups, metadata


# %%%% Helper functions for probabilistic_group_selection
def _calculate_landslide_proportion(
    group_probs, method="empirical", verbose: bool = False
):
    """
    Dynamically calculate an appropriate proportion of landslide groups.

    Parameters
    ----------
    group_probs : numpy.ndarray
        Mean failure probabilities per group (already aggregated)
    method : str, optional
        Method for proportion calculation

    Returns
    -------
    float
        Recommended proportion of landslide groups
    """
    valid_probs = group_probs[group_probs > 0]
    if len(valid_probs) == 0:
        return 0.0

    if verbose:
        print(
            f"Debug: Valid probs range: {np.min(valid_probs):.4f} to {np.max(valid_probs):.4f}"
        )
        print(f"Debug: Number of valid groups: {len(valid_probs)}")

    if method == "empirical":
        # Smoother scaling based on mean risk with continuous function
        mean_risk = np.mean(valid_probs)

        # Use sigmoid-like scaling for smoother transitions
        if mean_risk > 0.8:
            base_proportion = 0.15 + (mean_risk - 0.8) * 2.0  # 0.15 to 0.55
        elif mean_risk > 0.5:
            base_proportion = 0.10 + (mean_risk - 0.5) * 0.167  # 0.10 to 0.15
        elif mean_risk > 0.3:
            base_proportion = 0.08 + (mean_risk - 0.3) * 0.10  # 0.08 to 0.10
        else:
            base_proportion = 0.05 + mean_risk * 0.10  # 0.05 to 0.08

        # Add contribution from high-probability tail
        threshold = np.percentile(valid_probs, 80)
        high_prob_fraction = np.sum(valid_probs >= threshold) / len(valid_probs)

        # Blend base and tail-based proportions
        proportion = 0.7 * base_proportion + 0.3 * high_prob_fraction
        proportion = np.clip(proportion, 0.05, 0.7)

        if verbose:
            print(
                f"Debug: Empirical - mean_risk={mean_risk:.4f}, base={base_proportion:.4f}, "
                f"high_prob_frac={high_prob_fraction:.4f}, final={proportion:.4f}"
            )
        return proportion

    elif method == "statistical":
        mean_prob = np.mean(valid_probs)
        std_prob = np.std(valid_probs)
        q75 = np.percentile(valid_probs, 75)
        q90 = np.percentile(valid_probs, 90)

        # More stable threshold using percentiles with mean adjustment
        threshold = q75 + 0.5 * (q90 - q75)

        # Calculate proportion with minimum floor
        proportion = np.sum(valid_probs >= threshold) / len(valid_probs)

        # Adjust based on overall risk level
        risk_adjustment = np.clip(mean_prob / 0.5, 0.5, 2.0)
        proportion = proportion * risk_adjustment

        proportion = np.clip(proportion, 0.05, 0.65)
        if verbose:
            print(
                f"Debug: Statistical - threshold={threshold:.4f}, raw_prop={np.sum(valid_probs >= threshold) / len(valid_probs):.4f}, "
                f"risk_adj={risk_adjustment:.4f}, final={proportion:.4f}"
            )
        return proportion

    elif method == "risk_profile":
        mean_prob = np.mean(valid_probs)
        median_prob = np.median(valid_probs)
        std_prob = np.std(valid_probs)
        max_prob = np.max(valid_probs)
        q75 = np.percentile(valid_probs, 75)

        # Improved base score with more balanced weights
        cv = std_prob / mean_prob if mean_prob > 0 else 0  # Coefficient of variation

        base_score = (
            0.35 * (mean_prob / max_prob)
            + 0.25 * (median_prob / max_prob)
            + 0.20 * (q75 / max_prob)
            + 0.20 * np.clip(cv, 0, 1)
        )

        # Gentler, continuous scaling
        if mean_prob > 0.7:
            scaling_factor = 1.8
        elif mean_prob > 0.5:
            # Linear interpolation between 1.4 and 1.8
            scaling_factor = 1.4 + (mean_prob - 0.5) * 2.0
        elif mean_prob > 0.3:
            scaling_factor = 1.2 + (mean_prob - 0.3) * 1.0
        else:
            scaling_factor = 1.0 + mean_prob * 0.667

        proportion = np.clip(base_score * scaling_factor, 0.05, 0.75)
        if verbose:
            print(
                f"Debug: Risk profile - base_score={base_score:.4f}, cv={cv:.4f}, "
                f"scaling_factor={scaling_factor:.2f}, proportion={proportion:.4f}"
            )
        return proportion

    elif method == "adaptive":
        q25 = np.percentile(valid_probs, 25)
        q50 = np.percentile(valid_probs, 50)
        q75 = np.percentile(valid_probs, 75)
        iqr = q75 - q25
        mean_prob = np.mean(valid_probs)

        # More nuanced adaptive approach
        if iqr < 0.05:  # Very low variance - uniform risk
            proportion = 0.15 + (mean_prob * 0.4)
        elif iqr < 0.15:  # Low variance
            proportion = 0.12 + (mean_prob * 0.35) + (iqr * 0.5)
        elif iqr < 0.30:  # Moderate variance
            proportion = 0.10 + (mean_prob * 0.25) + (iqr * 0.8)
        else:  # High variance - diverse risk
            # Focus more on high-risk groups
            proportion = 0.08 + (q75 * 0.4) + (iqr * 0.3)

        proportion = np.clip(proportion, 0.05, 0.70)
        if verbose:
            print(
                f"Debug: Adaptive - mean={mean_prob:.4f}, median={q50:.4f}, "
                f"IQR={iqr:.4f}, proportion={proportion:.4f}"
            )
        return proportion

    elif method == "conservative":
        """New method: Conservative selection focusing on highest risk groups"""
        q90 = np.percentile(valid_probs, 90)
        q95 = np.percentile(valid_probs, 95)
        mean_prob = np.mean(valid_probs)

        # Focus on top groups, scale by overall risk
        threshold = 0.7 * q90 + 0.3 * q95
        base_proportion = np.sum(valid_probs >= threshold) / len(valid_probs)

        # Increase if overall risk is high
        if mean_prob > 0.6:
            proportion = base_proportion * 1.5
        elif mean_prob > 0.4:
            proportion = base_proportion * 1.2
        else:
            proportion = base_proportion

        proportion = np.clip(proportion, 0.05, 0.50)
        if verbose:
            print(
                f"Debug: Conservative - threshold={threshold:.4f}, proportion={proportion:.4f}"
            )
        return proportion

    else:
        raise ValueError(
            "Invalid method. Choose 'empirical', 'statistical', 'risk_profile', 'adaptive', or 'conservative'."
        )


# %% "pga_weighted" path

def generate_landslide_proportion_from_pga(
    grid,
    h_pga,
    v_pga,
    labeled_array,
    weight_array=None,
    slope_array=None,
    soil_condition_array=None,
    random_seed=None,
    verbose: bool = False,
):
    # --- reshape PGA to grid ---
    h_pga_array = np.asarray(h_pga).reshape(grid.shape)
    v_pga_array = np.asarray(v_pga).reshape(grid.shape)

    # --- optional arrays: accept 1-D (n_nodes) or 2-D (grid.shape) ---
    weight_grid = None
    if weight_array is not None:
        weight_arr = np.asarray(weight_array)
        if weight_arr.ndim == 1:
            if weight_arr.size != grid.number_of_nodes:
                raise ValueError(
                    f"weight_array length {weight_arr.size} != number_of_nodes {grid.number_of_nodes}"
                )
            weight_grid = weight_arr.reshape(grid.shape)
        elif weight_arr.shape == grid.shape:
            weight_grid = weight_arr
        else:
            raise ValueError("weight_array must be 1-D (n_nodes) or 2-D grid.shape")

    slope_grid = None
    if slope_array is not None:
        slope_arr = np.asarray(slope_array)
        slope_grid = slope_arr.reshape(grid.shape) if slope_arr.ndim == 1 else slope_arr

    soil_grid = None
    if soil_condition_array is not None:
        soil_arr = np.asarray(soil_condition_array)
        soil_grid = soil_arr.reshape(grid.shape) if soil_arr.ndim == 1 else soil_arr

    # --- seed for reproducibility (optional) ---
    if random_seed is not None:
        np.random.seed(random_seed)
        if verbose:
            print(f"Random seed = {random_seed}")

    unique_labels = np.unique(labeled_array)
    unique_labels = unique_labels[unique_labels != 0]

    probability_array = np.zeros_like(labeled_array, dtype=np.float32)
    group_probabilities = []
    metadata = {
        "group_data": [],
        "mean_h_pga": float(np.nanmean(h_pga_array)),
        "mean_v_pga": float(np.nanmean(v_pga_array)),
        "num_groups": int(len(unique_labels)),
    }

    # Early exit if no groups
    if len(unique_labels) == 0:
        return probability_array, 0.0, metadata

    for label_name in unique_labels:
        mask = labeled_array == label_name

        # group PGA values (may include NaNs at boundaries)
        group_h_pga = h_pga_array[mask]
        group_v_pga = v_pga_array[mask]

        # --- nan-safe means; guard all-NaN case ---
        mean_h_pga = np.nanmean(group_h_pga)
        mean_v_pga = np.nanmean(group_v_pga)
        if np.isnan(mean_h_pga):
            mean_h_pga = 0.0
        if np.isnan(mean_v_pga):
            mean_v_pga = 0.0

        # resultant and V/H ratio
        resultant_pga = float(np.sqrt(mean_h_pga**2 + mean_v_pga**2))
        vh_ratio = float(mean_v_pga / mean_h_pga) if mean_h_pga > 0 else 0.0

        # base probability from h-only and resultant
        h_prob = float(_calculate_prob_from_h_pga(mean_h_pga))
        r_prob = float(_calculate_prob_from_resultant(resultant_pga, vh_ratio))
        base_prob = 0.7 * h_prob + 0.3 * r_prob

        # --- define factors with defaults so they exist even if inputs are None ---
        slope_factor = 1.0
        soil_factor = 1.0
        weight_factor = 1.0

        if slope_grid is not None:
            group_slope = slope_grid[mask]
            mean_slope = float(np.nanmean(group_slope))
            slope_factor = float(_calculate_slope_factor(mean_slope))
            base_prob *= slope_factor

        if soil_grid is not None:
            group_soil = soil_grid[mask]
            mean_soil = float(np.nanmean(group_soil))
            soil_factor = float(0.5 + 0.5 * mean_soil)
            base_prob *= soil_factor

        if weight_grid is not None:
            group_weight = float(np.nanmean(weight_grid[mask]))
            epsilon = 1e-10
            weight_factor = float(1.0 / (group_weight + epsilon))
            base_prob *= weight_factor

        # stochastic variability
        stochastic_factor = float(np.random.lognormal(mean=0, sigma=0.3))
        group_prob = float(np.clip(base_prob * stochastic_factor, 0.0, 1.0))

        # write per-node probability for this group
        probability_array[mask] = group_prob
        group_probabilities.append(group_prob)

        # metadata for this group (factors always present)
        metadata["group_data"].append(
            {
                "label": int(label_name),
                "mean_h_pga": mean_h_pga,
                "mean_v_pga": mean_v_pga,
                "resultant_pga": resultant_pga,
                "vh_ratio": vh_ratio,
                "h_prob": h_prob,
                "r_prob": r_prob,
                "base_prob": float(base_prob),
                "slope_factor": slope_factor,
                "soil_factor": soil_factor,
                "weight_factor": weight_factor,
                "final_prob": group_prob,
            }
        )

    # global proportion = mean of group probs
    proportion = float(np.mean(group_probabilities))
    metadata["overall_proportion"] = proportion

    return probability_array, proportion, metadata



# %%% Region selection using probabilities

def select_groups_by_proportion_weighted(
    labeled_array,
    probability_array,
    proportion=None,
):
    """
    Selects a specified proportion of groups based on group-level probabilities.
    Robust to NaNs or zero-sum probability vectors by falling back to uniform selection.
    """
    unique_labels = np.unique(labeled_array)
    unique_labels = unique_labels[unique_labels != 0]
    num_groups = len(unique_labels)

    if num_groups == 0:
        return np.zeros_like(labeled_array), []

    # How many groups to select
    if proportion is not None:
        num_to_select = int(np.ceil(proportion * num_groups))
        num_to_select = max(1, min(num_to_select, num_groups))
    else:
        num_to_select = num_groups  # select all by probability

    # Per-group mean probability (nan-safe)
    group_probs = np.array(
        [float(np.nanmean(probability_array[labeled_array == lab])) for lab in unique_labels],
        dtype=float,
    )

    # Clean and normalize
    group_probs = np.nan_to_num(group_probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = group_probs.sum()

    if total <= 0.0:
        # Fallback: uniform probabilities if all zeros/NaNs
        p = np.ones(num_groups, dtype=float) / num_groups
    else:
        p = group_probs / total

    selected_labels = np.random.choice(unique_labels, num_to_select, replace=False, p=p)
    selected_groups = np.isin(labeled_array, selected_labels) * labeled_array
    return selected_groups, selected_labels.tolist()



# %%% Helper functions for "pga_weighted" path
def _calculate_prob_from_h_pga(h_pga):
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


def _calculate_prob_from_resultant(resultant_pga, vh_ratio):
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
    base_prob = _calculate_prob_from_h_pga(resultant_pga)

    # Modify based on V/H ratio - higher ratios can increase probability
    if vh_ratio > 0.5:  # Significant vertical component
        vh_factor = 1.0 + 0.2 * (vh_ratio - 0.5)
        base_prob *= min(vh_factor, 1.5)  # Cap the increase at 50%

    return base_prob


def _calculate_slope_factor(slope):
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
