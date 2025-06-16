"""
Functions for landslide selection
"""
# %% Required components
import numpy as np


# %% ### Region selection functions ###
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
def calculate_landslide_proportion_old(probability_array, method='empirical'):
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
    
    if len(valid_probs) == 0:
        return 0.0
    
    print(f"Debug: Valid probs range: {np.min(valid_probs):.4f} to {np.max(valid_probs):.4f}")
    print(f"Debug: Number of unique valid probabilities: {len(valid_probs)}")
    
    if method == 'empirical':
        # PROBLEM: This method returns probability values (0-1) as proportions
        # But proportions should represent fraction of groups to select
        
        # ORIGINAL (problematic):
        # proportion = np.percentile(valid_probs, 75)
        # normalized_proportion = proportion / np.max(valid_probs)
        
        # FIXED VERSION 1: Based on probability thresholds
        # Select groups with probabilities above 75th percentile
        threshold = np.percentile(valid_probs, 75)
        high_prob_groups = np.sum(probability_array >= threshold)
        total_groups = len(np.unique(probability_array[probability_array > 0]))
        proportion = high_prob_groups / len(np.unique(probability_array)) if len(np.unique(probability_array)) > 0 else 0
        
        # Alternative: Use a fixed reasonable proportion based on risk level
        # Higher risk = select more groups
        mean_risk = np.mean(valid_probs)
        if mean_risk > 0.7:
            proportion = 0.6  # High risk: select 60% of groups
        elif mean_risk > 0.5:
            proportion = 0.4  # Medium risk: select 40% of groups
        elif mean_risk > 0.3:
            proportion = 0.25  # Low-medium risk: select 25% of groups
        else:
            proportion = 0.15  # Low risk: select 15% of groups
        
        print(f"Debug: Empirical method - mean_risk={mean_risk:.4f}, proportion={proportion:.4f}")
        return proportion
    
    elif method == 'statistical':
        # PROBLEM: This calculates proportion of pixels, not proportion of groups
        
        # ORIGINAL (problematic):
        # mean_prob = np.mean(valid_probs)
        # std_prob = np.std(valid_probs)
        # threshold = mean_prob + std_prob
        # proportion = np.sum(probability_array >= threshold) / probability_array.size
        
        # FIXED VERSION: Calculate proportion of groups above threshold
        mean_prob = np.mean(valid_probs)
        std_prob = np.std(valid_probs)
        threshold = mean_prob + std_prob
        
        # Count unique groups (not pixels) above threshold
        unique_probs_above_threshold = valid_probs[valid_probs >= threshold]
        proportion = len(unique_probs_above_threshold) / len(valid_probs)
        
        # Ensure minimum selection
        proportion = max(proportion, 0.1)  # At least 10% of groups
        
        print(f"Debug: Statistical method - threshold={threshold:.4f}, proportion={proportion:.4f}")
        return proportion
    
    elif method == 'risk_profile':
        # This method has the right idea but values are often too small
        
        mean_prob = np.mean(valid_probs)
        median_prob = np.median(valid_probs)
        std_prob = np.std(valid_probs)
        max_prob = np.max(valid_probs)
        
        # ORIGINAL (often too small):
        # risk_score = (
        #     0.4 * (mean_prob / max_prob) + 
        #     0.3 * (median_prob / max_prob) + 
        #     0.3 * np.clip(std_prob / mean_prob, 0, 1)
        # )
        
        # IMPROVED VERSION: More aggressive scaling
        base_score = (
            0.4 * (mean_prob / max_prob) + 
            0.3 * (median_prob / max_prob) + 
            0.3 * np.clip(std_prob / mean_prob, 0, 1)
        )
        
        # Apply scaling factor based on overall risk level
        if mean_prob > 0.6:
            scaling_factor = 2.0
        elif mean_prob > 0.4:
            scaling_factor = 1.5
        else:
            scaling_factor = 1.2
            
        proportion = np.clip(base_score * scaling_factor, 0.1, 0.8)
        
        print(f"Debug: Risk profile method - base_score={base_score:.4f}, scaling_factor={scaling_factor:.2f}, proportion={proportion:.4f}")
        return proportion
    
    elif method == 'adaptive':
        # NEW METHOD: Adaptive based on probability distribution shape
        
        # Analyze distribution characteristics
        q25 = np.percentile(valid_probs, 25)
        q75 = np.percentile(valid_probs, 75)
        iqr = q75 - q25
        mean_prob = np.mean(valid_probs)
        
        # If probabilities are tightly clustered (small IQR), select fewer groups
        # If probabilities are spread out (large IQR), select more groups
        if iqr < 0.1:
            # Tight distribution - select based on mean
            proportion = 0.2 + (mean_prob * 0.3)
        else:
            # Spread distribution - select more groups
            proportion = 0.3 + (iqr * 0.5)
        
        proportion = np.clip(proportion, 0.15, 0.7)
        
        print(f"Debug: Adaptive method - mean={mean_prob:.4f}, IQR={iqr:.4f}, proportion={proportion:.4f}")
        return proportion
    
    else:
        raise ValueError("Invalid method. Choose 'empirical', 'statistical', 'risk_profile', or 'adaptive'.")

def probabilistic_group_selection_old(labeled_array, probability_array, proportion_method='empirical', 
                                custom_proportion=None, random_seed=5000):
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
        Seed for reproducibility, default: 5000
    
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

def probabilistic_group_selection(labeled_array, probability_array, proportion_method='empirical', 
                                  custom_proportion=None, random_seed=5000):
    """
    Enhanced probabilistic group selection with dynamic proportion calculation.
    """
    
    unique_labels = np.unique(labeled_array)
    unique_labels = unique_labels[unique_labels != 0]
    num_groups = len(unique_labels)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    print(f"Debug: Total groups available: {num_groups}")
    
    # Determine proportion
    if custom_proportion is not None:
        proportion = custom_proportion
        method = 'user_defined'
        print(f"Debug: Using custom proportion: {proportion}")
    else:
        proportion = calculate_landslide_proportion(
            probability_array, 
            method=proportion_method
        )
        method = proportion_method
    
    # Get probabilities for each group
    group_probs = []
    group_mean_probs = []  # Store actual probability values for debugging
    for label_name in unique_labels:
        mask = (labeled_array == label_name)
        group_prob = np.mean(probability_array[mask])
        group_probs.append(group_prob)
        group_mean_probs.append(group_prob)
    
    group_probs = np.array(group_probs)
    print(f"Debug: Group probability range: {np.min(group_probs):.4f} to {np.max(group_probs):.4f}")
    
    # Handle edge case where all probabilities are zero
    if np.sum(group_probs) == 0:
        print("Warning: All group probabilities are zero!")
        return np.zeros_like(labeled_array), 0.0
    
    # Normalize probabilities for selection
    normalized_probs = group_probs / np.sum(group_probs)
    
    # Calculate number to select
    num_to_select = max(1, int(np.ceil(num_groups * proportion)))  # Ensure at least 1
    num_to_select = min(num_to_select, num_groups)  # Don't exceed available groups
    
    print(f"Debug: Proportion={proportion:.4f}, Selecting {num_to_select} out of {num_groups} groups")
    
    # Select groups based on probabilities
    selected_labels = np.random.choice(
        unique_labels, num_to_select, replace=False, p=normalized_probs)
    
    selected_groups = np.isin(labeled_array, selected_labels) * labeled_array
    
    # Enhanced metadata
    metadata = {
        'method_used': method,
        'proportion_calculated': proportion,
        'num_groups_total': num_groups,
        'num_groups_selected': num_to_select,
        'selected_labels': selected_labels.tolist(),
        'group_probabilities': dict(zip(unique_labels.tolist(), group_mean_probs)),
        'selection_probabilities': dict(zip(unique_labels.tolist(), normalized_probs.tolist()))
    }
    
    return selected_groups, metadata

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
        h_prob = _calculate_prob_from_h_pga(mean_h_pga)
        r_prob = _calculate_prob_from_resultant(resultant_pga, vh_ratio)
        base_prob = 0.7 * h_prob + 0.3 * r_prob
        
        # Apply slope factor if available
        slope_factor = 1.0
        if slope_array is not None:
            group_slope = slope_array[mask]
            mean_slope = np.mean(group_slope)
            slope_factor = _calculate_slope_factor(mean_slope)
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