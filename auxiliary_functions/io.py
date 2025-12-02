"""
auxiliary_functions/io.py

Functions that allow import and output of data
"""

# %% Required packages

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy
import pickle
import pandas as pd
import geopandas as gpd
from .stats import fit_bivariate_kde

# %% Handle JSON config files
# %%% Main function
def get_config(config_input: Optional[Union[str, Path, Dict[str, Any]]] = None,
            merge_with_defaults: bool = True,
            validate: bool = True) -> Dict[str, Any]:
    """
    Get configuration from various input types.
    
    Args:
        config_input: Configuration input. Can be:
            - None: Returns default configuration
            - str/Path: Path to JSON configuration file
            - dict: Configuration dictionary
        merge_with_defaults (bool): If True, merge loaded/provided config with defaults
                                    to ensure all required keys are present.
        validate (bool): If True, validate the configuration before returning.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        ValueError: If configuration validation fails
        TypeError: If config_input is not a supported type
        
    Example:
        # Use default configuration
        config = get_config()
        
        # Load from JSON file
        config = get_config('my_config.json')
        
        # Use dictionary (merged with defaults)
        config = get_config({'dem_info': {'north': 30.0}})
        
        # Use dictionary without merging defaults
        config = get_config(my_complete_dict, merge_with_defaults=False)
    """
    if config_input is None:
        # Return default configuration
        config = get_default_config()
        
    elif isinstance(config_input, (str, Path)):
        # Load from JSON file
        loaded_config = load_config_from_json(config_input)
        
        if merge_with_defaults:
            # Merge with defaults to ensure all keys are present
            default_config = get_default_config()
            config = merge_configs(default_config, loaded_config)
        else:
            config = loaded_config
            
    elif isinstance(config_input, dict):
        # Handle dictionary input
        if merge_with_defaults:
            # Merge with defaults to ensure all keys are present
            default_config = get_default_config()
            config = merge_configs(default_config, config_input)
        else:
            config = config_input.copy()  # Make a copy to avoid modifying original
            
    else:
        raise TypeError(f"config_input must be dict, str, Path, or None. Got {type(config_input)}")
    
    # Validate configuration before returning
    if validate:
        validate_config(config)
    
    return config

# %%% Helper functions
def get_default_config() -> Dict[str, Any]:
    """
    Returns the default configuration dictionary.
    
    Returns:
        Dict[str, Any]: Default configuration parameters
    """
    return {
        'dem_info': {
            'dem_type': "SRTMGL1",
            'north': 28.29, #31.34,# 28.29,
            'east': 85.20, # 85.00, #103.70,
            'south': 28.18, #31.23, # 28.18,
            'west': 85.04, # 84.84, #103.56,
            'buffer': 0.01,
            'smooth_num': 4,
            'plot_dem' : True
            },
        'flow_params': {
            'flow_metric': 'D8',
            'separate_hill_flow': True,
            'depression_handling': 'fill',
            'update_hill_depressions': True,
            'accumulate_flow': True
            },
        'soil_params': {
            'angle_int_frict': np.radians(30),
            'cohesion_eff': 15e3,  # Pa
            'submerged_soil_proportion': 0.5,
            'max_soil_depth': 1.5, # m
            'distribution': 'elevation', # 'uniform' or 'elevation'
            'relationship': 'exponential', # 'linear', 'exponential', 'power', 'logarithmic', 'sigmoid'
            'decay_rate': 5.0, # rate of decay of exponential function
            'exponent': 1.0, # exponent for when relationship == 'power'
            'drainage_transform': 'log', # transformation for drainage area values
            'drainage_threshold': None, # drainage area threshold when  'drainage_transform'=='threshold'
            'plot_soil': True,
            },
        'pga': {
            'horizontal_max': 0.6,
            'vertical_max': 0.2,
            'distribution': "uniform",
            'plot_grids': False
            },
        'simulation': {
            'time_shaking': 10,  # seconds
            'displacement_threshold': 0,
            'aspect_interval': 20,
            'random_seed': 5000, # for reproducibility
            'handle_small_regions': 'merge', # what happens to 1px regions: 'keep', 'merge', or 'remove'
            'split_convergence': 0.75, # threshold for splitting iterations
            'min_region_size': 10, # minimum size of region to split
            'selection_method': 'probabilistic', # or 'pga_weighted'
            'proportion_method': 'statistical', # 'empirical', 'statistical', 'risk_profile', or 'adaptive'
            },
        'plot_intermediates':{
            'factor_of_safety': False,
            'critical_acceleration': False,
            'unstable_areas': False, # Issue here
            'filled_and_split': True
            },
        "output": {
            "save_plots": False,
            "output_dir": None,     # defaults to current directory
            "save_pickle": True,
            "load_pickle": True,
            },
        }


def load_config_from_json(json_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        json_file (Union[str, Path]): Path to the JSON configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary loaded from JSON
        
    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
    """
    json_path = Path(json_file)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        # Convert angle_int_frict back to radians if it was stored as degrees
        if 'soil_params' in config and 'angle_int_frict' in config['soil_params']:
            # Check if the value seems to be in degrees (> pi would indicate degrees)
            angle_val = config['soil_params']['angle_int_frict']
            if angle_val > np.pi:
                config['soil_params']['angle_int_frict'] = np.radians(angle_val)
        
        return config
        
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error parsing JSON file {json_path}: {e}")


def save_config_to_json(config: Dict[str, Any], json_file: Union[str, Path]) -> None:
    """
    Save configuration dictionary to a JSON file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary to save
        json_file (Union[str, Path]): Path where to save the JSON file
    """
    json_path = Path(json_file)
    
    # Create a copy to avoid modifying the original
    config_copy = copy.deepcopy(config)
    
    # Convert radians to degrees for better human readability in JSON
    if 'soil_params' in config_copy and 'angle_int_frict' in config_copy['soil_params']:
        config_copy['soil_params']['angle_int_frict'] = np.degrees(
            config['soil_params']['angle_int_frict']
        )
    
    # Create directory if it doesn't exist
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(config_copy, f, indent=4)


def merge_configs(default_config: Dict[str, Any], 
                user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge user configuration with default configuration.
    User values override defaults.
    
    Args:
        default_config (Dict[str, Any]): Default configuration
        user_config (Dict[str, Any]): User-provided configuration
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    merged = copy.deepcopy(default_config)
    
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Basic validation of configuration parameters.
    
    Args:
        config (Dict[str, Any]): Configuration to validate
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['dem_info', 'flow_params', 'soil_params', 'pga', 
                        'simulation', 'plot_intermediates', 'output']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate specific parameters
    dem_info = config['dem_info']
    if dem_info['north'] <= dem_info['south']:
        raise ValueError("North coordinate must be greater than south coordinate")
    
    if dem_info['east'] <= dem_info['west']:
        raise ValueError("East coordinate must be greater than west coordinate")
    
    soil_params = config['soil_params']
    if soil_params['angle_int_frict'] < 0 or soil_params['angle_int_frict'] > np.pi/2:
        raise ValueError("Angle of internal friction must be between 0 and π/2 radians")
    
    if soil_params['cohesion_eff'] < 0:
        raise ValueError("Effective cohesion must be non-negative")
    
    pga = config['pga']
    if pga['horizontal_max'] < 0 or pga['vertical_max'] < 0:
        raise ValueError("PGA values must be non-negative")
    
    simulation = config['simulation']
    if simulation['time_shaking'] <= 0:
        raise ValueError("Shaking time must be positive")
    
    return True

# %% Pickling datasets

# %% Import measured data
# # Import area, length and width data for all measured landslides in region
# file_name = "C:/Users/sghoshal/Documents/ArcGIS/Projects/landslides_Nepal/measuredLandslides_all.csv"
# measured_data = pd.read_csv(file_name)

# # Import zonal statistics for Roback et al. 2017 landslides
# file_name2 = "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_spatialStats.csv"
# file_name3 = "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_ZonalStats_clipbuffer.csv"
# measured_spatial_stats = pd.read_csv(file_name2)
# measured_spatial_stats_clipped = pd.read_csv(file_name3)

# # Remove all landslides below 1000 m^2
# measured_spatial_stats_900greater = measured_spatial_stats.drop(measured_spatial_stats[measured_spatial_stats['Area']<1000].index)

# plot_order = ["Roback2017_Gorkha", "Jones2021_ASM"]

# # Import Roback et al. 2017 landslide shapefile for test area
# LSshapefile_name = 'C:/Users/sghoshal/Documents/ArcGIS/Projects/landslides_Nepal/landslide_Nepal_Roback.shp'
# LSshapefile_file = gpd.read_file(LSshapefile_name)

# Parameters included in filenames
FILENAME_PARAMS = [
    "dem_type",
    "cohesion_eff",
    "angle_int_frict",
    "distribution",
    "relationship",
    "curvature_variant",
    "random_seed",
]

# Define which parameters should only be included if they're not None/empty
OPTIONAL_PARAMS = {
    "relationship",
    "curvature_variant", 
    "random_seed",
}

# Parameter abbreviations
PARAM_ABBREVIATIONS = {
    "dem_type": "dem",
    "cohesion_eff": "c",
    "distribution": "dist",
    "relationship": "rel",
    "curvature_variant": "curv",
    "random_seed": "seed",
    "angle_int_frict": "intfr"
}

# Reverse mapping for parsing
REVERSE_ABBREVIATIONS = {v: k for k, v in PARAM_ABBREVIATIONS.items()}

def pickle_or_not_to_pickle(file_name_dict, pickle_path="measured_data.pkl"):
    """
    Load processed data (DataFrames, shapefile, KDEs) from pickle if it exists.
    Otherwise, build from source files, save, and return.
    """
    
    if os.path.exists(pickle_path):
        print(f"Loading preprocessed data from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            bundle = pickle.load(f)
        return bundle
    
    print("Pickle not found, building from CSVs and shapefile...")

    # --- Load CSVs ---
    # All measured landslide areas
    measured_data = pd.read_csv(file_name_dict['file1'])
    
    # All measured landslide zonal statistics (elevation, slope, aspect)
    measured_spatial_stats = pd.read_csv(file_name_dict['file2'])
    
    # Filter out landslides below sensitivity threshold
    measured_spatial_stats_900greater = measured_spatial_stats.drop(
        measured_spatial_stats[measured_spatial_stats['Area'] <= 900].index
    )
    
    # Measured landslide zonal statistics inside selected area
    measured_spatial_stats_clipped = pd.read_csv(file_name_dict['file3'])

    # --- Load shapefile ---
    LSshapefile_file = gpd.read_file(file_name_dict['shapefile_name'])

    # --- Fit KDE ---
    kde_data, kde_transform = fit_bivariate_kde(
        dataframe=measured_data,
        x_col="length_m",
        y_col="width_m",
        category_col=None,
        plot_results=False
    )

    # --- Bundle everything ---
    bundle = {
        "measured_data": measured_data,
        "measured_spatial_stats": measured_spatial_stats,
        "measured_spatial_stats_clipped": measured_spatial_stats_clipped,
        "measured_spatial_stats_900greater": measured_spatial_stats_900greater,
        "LSshapefile_file": LSshapefile_file,
        "kde_data": kde_data,
        "kde_transform": kde_transform
    }

    # Save to pickle for next time
    with open(pickle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved preprocessed data to {pickle_path}")

    return bundle

def parse_pickle_name_new(file_name: str) -> Dict[str, Any]:
    """
    Parse key=value pickle filename back into parameter dict.
    Handles any order and missing optional parameters gracefully.
    
    Example: 
        "dem=synthetic_c=5_dist=elevation_rel=linear.pkl"
        -> {"dem_type": "synthetic", "cohesion_eff": 5, ...}
    
    ANALYSIS FUNCTION - called by load_all_runs()
    """
    base = os.path.splitext(os.path.basename(file_name))[0]
    parts = base.split("_")
    
    # Initialize with None for all parameters
    params = {param: None for param in FILENAME_PARAMS}
    
    # Parse each key=value pair
    for part in parts:
        if "=" not in part:
            print(f"Warning: Skipping malformed part '{part}' in {file_name}")
            continue
        
        key, value = part.split("=", 1)
        param_name = REVERSE_ABBREVIATIONS.get(key, key)
        
        if param_name not in params:
            print(f"Warning: Unknown parameter '{param_name}' in {file_name}")
            continue
        
        # Type conversion
        if param_name == "cohesion_eff":
            params[param_name] = int(value)
        elif param_name == "random_seed" and value != "None":
            params[param_name] = int(value)
        else:
            params[param_name] = value if value != "None" else None
    
    return params

def make_key_new(params: Dict[str, Any]) -> tuple:
    """
    Create a tuple key from params dictionary.
    Uses all defined filename parameters in order.
    
    ANALYSIS FUNCTION - called by load_all_runs()
    """
    return tuple(params.get(param) for param in FILENAME_PARAMS)

def parse_pickle_name(file_name):
    """
    Parse deterministic pickle filename back into parameter dict.
    Handles variable filename structures with additional components.
    
    Special handling for:
    - 'drainage_area' as single distribution type
    - 'std_global'/'std_local' as curvature variants
    """
    base = os.path.splitext(file_name)[0]
    parts = base.split("_")
    
    dem = parts[0]
    coh = int(parts[1][1:])  # strip 'c'
    
    params = {
        "dem_type": dem,
        "cohesion_eff": coh,
        "distribution": None,
        "relationship": None,
        "curvature_variant": None,  # New field for std_global/std_local
        "random_seed": None,
    }
    
    # Find seed first (it's always at the end if present)
    seed_idx = None
    for i, part in enumerate(parts):
        if part.startswith("seed"):
            params["random_seed"] = int(part[4:])
            seed_idx = i
            break
    
    # Handle special case: drainage_area
    if len(parts) > 3 and parts[2] == "drainage" and parts[3] == "area":
        params["distribution"] = "drainage_area"
        return params
    
    # Standard distribution
    params["distribution"] = parts[2]
    
    # Handle relationship and curvature variants
    if params["distribution"] in ("elevation", "curvature"):
        idx = 3
        # Look for relationship
        if idx < len(parts) and parts[idx] in ("linear", "exponential"):
            params["relationship"] = parts[idx]
            idx += 1
            
            # For curvature with linear, check for std variants
            if params["distribution"] == "curvature" and params["relationship"] == "linear":
                if idx < len(parts) and parts[idx] == "std":
                    idx += 1  # Move past "std"
                    if idx < len(parts) and parts[idx] in ("global", "local"):
                        params["curvature_variant"] = f"std_{parts[idx]}"
    
    return params

def make_key(params):
    """
    Create a tuple key from params.
    Now includes curvature_variant as 5th element.
    Structure: (cohesion, distribution, relationship, curvature_variant, seed)
    """
    return (
        params["cohesion_eff"],
        params["distribution"],
        params["relationship"],      # can be None
        params["curvature_variant"], # can be None
        params["random_seed"],       # can be None
    )

def load_all_runs_new(folder_path: str) -> Dict[tuple, Any]:
    """
    Load all pickle files in a folder and store them in a dictionary.
    Keys: parameter tuples with values in FILENAME_PARAMS order
    Values: run data
    
    ANALYSIS FUNCTION - main entry point for loading saved runs
    """
    runs_dict = {}
    run_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    
    print("Loading pickle files:")
    print("=" * 60)
    
    for file_name in run_files:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            with open(file_path, "rb") as f:
                run_data = pickle.load(f)
            
            params = parse_pickle_name(file_name)
            key = make_key(params)
            runs_dict[key] = run_data
            
            print(f"File: {file_name}")
            print(f"  Parsed params: {params}")
            print(f"  Key: {key}")
            print()
        except Exception as e:
            print(f"Error loading {file_name}: {e}")
            print()
    
    return runs_dict

def load_all_runs(folder_path):
    """
    Load all pickle files in a folder and store them in a dictionary.
    Keys: parameter tuples (cohesion, distribution, relationship, curvature_variant, seed)
    Values: run data
    """
    runs_dict = {}
    run_files = [f for f in os.listdir(folder_path) if f.endswith(".pkl")]
    
    print("Loading pickle files:")
    print("=" * 60)
    
    for file_name in run_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "rb") as f:
            run_data = pickle.load(f)
        
        params = parse_pickle_name(file_name)
        key = make_key(params)
        runs_dict[key] = run_data
        
        print(f"File: {file_name}")
        print(f"  Parsed params: {params}")
        print(f"  Key: {key}")
        print()
    
    return runs_dict

def filter_runs(runs_dict: Dict[tuple, Any], **filters) -> Dict[tuple, Any]:
    """
    Filter runs by parameter values.
    
    ANALYSIS FUNCTION - helper for selecting specific runs
    
    Example:
        filter_runs(runs_dict, cohesion_eff=5, distribution="elevation")
    """
    filtered = {}
    
    for key, data in runs_dict.items():
        params = dict(zip(FILENAME_PARAMS, key))
        
        # Check if all filter conditions match
        if all(params.get(k) == v for k, v in filters.items()):
            filtered[key] = data
    
    return filtered