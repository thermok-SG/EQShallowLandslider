"""
auxiliary_functions/io.py

Functions that allow import and output of data
"""

# %% Required packages

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy

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
            'north': 28.29,
            'east': 85.20,
            'south': 28.18,
            'west': 85.04,
            'buffer': 0.01,
            'smooth_num': 4,
            'plot_dem': True
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
            'max_soil_depth': 1.0,  # m
            'distribution': 'uniform',
            'plot_soil': False,
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
            'selection_method': 'probabilistic', # or 'pga_weighted'
            'proportion_method': 'statistical', # 'empirical', 'statistical', 'risk_profile', or 'adaptive'
        },
        'plot_intermediates': {
            'factor_of_safety': False,
            'critical_acceleration': False,
            'unstable_areas': False,
            'filled_and_split': True
        },
        'output': {
            'save_plots': False,
            'output_dir': None,
        }
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