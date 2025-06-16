# -*- coding: utf-8 -*-
"""
Landslide modeling class based on Newmark displacement method.

This class implements:
1. Regions of instability calculation
2. Region filtering by aspect
3. Newmark displacement calculation for each pixel
4. Sediment transport using multi-flow approach

@author: sghoshal
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import binary_fill_holes, generate_binary_structure
from tqdm import tqdm
import yaml
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any

# Landlab components
from landlab import RasterModelGrid
from landlab.io import esri_ascii
from landlab.components import PriorityFloodFlowRouter

# Import auxiliary functions (assuming they exist)
from auxiliary_functions import (
    get_topo, smooth_elevation_grid, generate_acceleration_grid, apply_soil_depth,
    critical_transient_acceleration, factor_of_safety, calculate_regions,
    calculate_region_properties, split_groups_by_aspect, create_zones,
    split_wide_regions, generate_landslide_probability, probabilistic_group_selection,
    generate_landslide_proportion_from_pga, select_groups_by_proportion_weighted,
    calculate_newmark_displacement, trace_paths_landslides, update_soil_depth,
    fit_bivariate_kde
)


class LandslideModel:
    """
    A comprehensive landslide modeling class using Newmark displacement method.
    
    This class handles the complete workflow from DEM processing to sediment transport
    simulation, with configurable parameters and multiple analysis methods.
    """
    
    def __init__(self, config: Optional[Union[Dict, str, Path]] = None):
        """
        Initialize the LandslideModel.
        
        Parameters
        ----------
        config : dict, str, Path, or None
            Configuration dictionary, path to YAML file, or None for defaults
        """
        self.config = self._load_config(config)
        self._validate_config()
        
        # Initialize state variables
        self.grid = None
        self.z = None
        self.slopes = None
        self.slopes_degrees = None
        self.aspect_nodes = None
        self.soil_depth = None
        self.flow_router = None
        
        # Analysis results
        self.factor_of_safety_vals = None
        self.acceleration_arrays = {}
        self.critical_accelerations = {}
        self.instability_regions = {}
        self.selected_groups = {}
        self.newmark_displacement = {}
        self.landslide_paths = {}
        self.updated_soil_depth = None
        
        # Analysis metadata
        self.analysis_metadata = {}
        
    def _load_config(self, config: Optional[Union[Dict, str, Path]]) -> Dict:
        """Load configuration from various sources."""
        if config is None:
            return self._default_config()
        elif isinstance(config, dict):
            return self._merge_with_defaults(config)
        elif isinstance(config, (str, Path)):
            return self._load_yaml_config(config)
        else:
            raise ValueError("Config must be dict, string path, Path object, or None")
    
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'dem_info': {
                'dem_type': "SRTMGL1",
                'north': 28.29,
                'east': 85.20,
                'south': 28.18,
                'west': 85.04,
                'buffer': 0.01,
                'smooth_num': 4,
                'dem_file': None  # Path to local DEM file
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
                'g': 9.81  # m/s^2
            },
            'pga': {
                'horizontal': 0.6,
                'vertical': 0.2,
                'distribution': "uniform"
            },
            'simulation': {
                'time_shaking': 10,  # seconds
                'displacement_threshold': 0,
                'aspect_interval': 20,
                'min_region_size': 2,
                'width_threshold': 2.0
            },
            'analysis_methods': {
                'use_probabilistic': True,
                'use_pga_weighted': True,
                'split_wide_regions': True,
                'trace_sediment_paths': True
            },
            'output': {
                'save_results': False,
                'output_dir': None,
                'include_intermediate': True
            }
        }
    
    def _merge_with_defaults(self, user_config: Dict) -> Dict:
        """Merge user config with defaults."""
        default = self._default_config()
        
        def deep_merge(default_dict, user_dict):
            result = default_dict.copy()
            for key, value in user_dict.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(default, user_config)
    
    def _load_yaml_config(self, config_path: Union[str, Path]) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        return self._merge_with_defaults(user_config)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Basic validation - extend as needed
        required_keys = ['dem_info', 'soil_params', 'pga', 'simulation']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config section: {key}")
    
    def load_dem(self) -> Tuple[RasterModelGrid, np.ndarray]:
        """
        Load and prepare DEM data.
        
        Returns
        -------
        grid : RasterModelGrid
            Landlab grid object
        z : np.ndarray
            Elevation values
        """
        dem_config = self.config['dem_info']
        
        if dem_config.get('dem_file'):
            # Load from file
            with open(dem_config['dem_file']) as fp:
                self.grid = esri_ascii.load(fp, name="topographic__elevation", at="node")
            self.z = self.grid.at_node["topographic__elevation"]
        else:
            # Download DEM
            self.grid, self.z = get_topo(
                dem_type=dem_config['dem_type'],
                north=dem_config['north'],
                south=dem_config['south'],
                east=dem_config['east'],
                west=dem_config['west'],
                buffer=dem_config['buffer']
            )
            
            # Smooth downloaded DEM
            if dem_config['smooth_num'] > 0:
                smoothed_elev = smooth_elevation_grid(
                    self.grid, 
                    method='gaussian', 
                    smooth_num=dem_config['smooth_num']
                )
                self.grid.at_node["topographic__elevation"] = smoothed_elev
        
        # Open all boundary nodes
        self.grid.set_closed_boundaries_at_grid_edges(
            right_is_closed=False, top_is_closed=False,
            left_is_closed=False, bottom_is_closed=False
        )
        
        return self.grid, self.z
    
    def setup_soil_properties(self):
        """Setup soil depth and bedrock elevation."""
        soil_config = self.config['soil_params']
        
        # Apply soil depth
        self.soil_depth = apply_soil_depth(
            self.grid, 
            max_soil_depth=soil_config['max_soil_depth'],
            distribution=soil_config['distribution']
        )
        
        # Add bedrock elevation field
        self.grid.add_zeros("bedrock__elevation", at="node", clobber=True)
        self.grid.at_node["bedrock__elevation"][:] = self.grid.at_node["topographic__elevation"]
        self.grid.at_node["topographic__elevation"][:] += self.grid.at_node["soil__depth"]
    
    def calculate_slopes_and_aspects(self):
        """Calculate slope and aspect arrays."""
        self.slopes = self.grid.calc_slope_at_node(elevs='topographic__elevation')
        self.slopes_degrees = np.degrees(self.slopes)
        
        self.aspect_nodes = np.array(
            self.grid.calc_aspect_at_node(
                elevs='topographic__elevation', 
                unit="degrees",
                ignore_closed_nodes=True
            )
        )
        self.aspect_nodes[self.grid.boundary_nodes] = np.nan
        
        # Store statistics
        self.analysis_metadata['slopes'] = {
            'mean': np.nanmean(self.slopes_degrees),
            'std': np.nanstd(self.slopes_degrees),
            'median': np.nanmedian(self.slopes_degrees)
        }
    
    def initialize_flow_router(self):
        """Initialize and run flow router."""
        flow_config = self.config['flow_params']
        
        self.flow_router = PriorityFloodFlowRouter(
            self.grid,
            flow_metric=flow_config['flow_metric'],
            separate_hill_flow=flow_config['separate_hill_flow'],
            depression_handler=flow_config['depression_handling'],
            update_hill_depressions=flow_config['update_hill_depressions'],
            accumulate_flow=flow_config['accumulate_flow']
        )
        
        self.flow_router.run_one_step()
    
    def generate_acceleration_grids(self):
        """Generate PGA arrays for horizontal and vertical acceleration."""
        pga_config = self.config['pga']
        
        h_array, v_array = generate_acceleration_grid(
            self.grid,
            max_horizontal=pga_config['horizontal'],
            max_vertical=pga_config['vertical'],
            distribution=pga_config['distribution']
        )
        
        self.acceleration_arrays = {
            'horizontal': h_array,
            'vertical': v_array
        }
        
        return self.acceleration_arrays
    
    def calculate_stability_analysis(self):
        """Perform complete stability analysis."""
        soil_config = self.config['soil_params']
        
        # Calculate factor of safety
        self.factor_of_safety_vals = factor_of_safety(
            self.grid, 
            soil_config['cohesion_eff'], 
            soil_config['angle_int_frict']
        )
        
        # Calculate critical accelerations
        g = soil_config['g']
        
        # Without earthquake
        a_transient_zero, a_sliding_zero, a_diff_zero = critical_transient_acceleration(
            self.grid, 
            soil_config['cohesion_eff'], 
            soil_config['angle_int_frict'],
            submerged_soil_proportion=soil_config['submerged_soil_proportion'], 
            a_h=0, a_v=0
        )
        
        # With earthquake
        a_transient_EQ, a_sliding_EQ, a_diff_EQ = critical_transient_acceleration(
            self.grid, 
            soil_config['cohesion_eff'], 
            soil_config['angle_int_frict'],
            submerged_soil_proportion=soil_config['submerged_soil_proportion'],
            a_h=self.acceleration_arrays['horizontal'] * g,
            a_v=self.acceleration_arrays['vertical'] * g
        )
        
        self.critical_accelerations = {
            'no_earthquake': {
                'transient': a_transient_zero,
                'sliding': a_sliding_zero,
                'difference': a_diff_zero
            },
            'with_earthquake': {
                'transient': a_transient_EQ,
                'sliding': a_sliding_EQ,
                'difference': a_diff_EQ
            }
        }
        
        # Identify unstable locations
        sliding_locations_bool = a_sliding_EQ > a_transient_EQ
        self.instability_regions['sliding_locations'] = sliding_locations_bool
        
        return self.critical_accelerations
    
    def identify_failure_regions(self):
        """Identify and process failure regions."""
        sliding_locations_bool = self.instability_regions['sliding_locations']
        sliding_locations_bool[self.grid.boundary_nodes] = False
        sliding_array_bool = sliding_locations_bool.reshape(self.grid.shape)
        
        # Calculate regions
        labeled_array, num_features = calculate_regions(sliding_array_bool, connect_val=8)
        
        # Fill holes
        struct = generate_binary_structure(2, 2)
        labeled_groups = np.unique(labeled_array)
        labeled_array_filled = np.zeros_like(labeled_array)
        
        for label_number in tqdm(labeled_groups, desc="Filling regions"):
            if label_number > 0:
                masked_array = ~np.ma.getmask(
                    np.ma.masked_not_equal(labeled_array, label_number)
                )
                labeled_array_filled[binary_fill_holes(masked_array, structure=struct)] = label_number
        
        # Split by aspect
        aspect_intervals = create_zones(interval=self.config['simulation']['aspect_interval'])
        aspect_nodes_array = self.aspect_nodes.reshape(self.grid.shape)
        
        aspect_subgroups, aspect_zones, info = split_groups_by_aspect(
            labeled_array_filled, aspect_nodes_array, zones=aspect_intervals
        )
        
        self.instability_regions.update({
            'labeled_array': labeled_array,
            'labeled_array_filled': labeled_array_filled,
            'aspect_subgroups': aspect_subgroups,
            'aspect_zones': aspect_zones,
            'num_features': num_features
        })
        
        return aspect_subgroups
    
    def select_landslide_groups(self, method: str = 'probabilistic'):
        """
        Select landslide groups using specified method.
        
        Parameters
        ----------
        method : str
            Selection method: 'probabilistic' or 'pga_weighted'
        """
        if method == 'probabilistic':
            return self._probabilistic_selection()
        elif method == 'pga_weighted':
            return self._pga_weighted_selection()
        else:
            raise ValueError(f"Unknown selection method: {method}")
    
    def _probabilistic_selection(self):
        """Select groups using probabilistic method."""
        probabilities, prob_metadata = generate_landslide_probability(
            self.grid,
            h_pga_array=self.acceleration_arrays['horizontal'],
            v_pga_array=self.acceleration_arrays['vertical'],
            labeled_array=self.instability_regions['aspect_subgroups'],
            slope_array=self.slopes_degrees,
            soil_array=None,
            geological_factor_array=None,
            critical_acceleration_array=self.critical_accelerations['no_earthquake']['transient'],
            default_critical_acceleration=0.2,
            random_seed=None,
            normalise_final_probs=True
        )
        
        selected_groups, selected_proportion = probabilistic_group_selection(
            probability_array=probabilities,
            labeled_array=self.instability_regions['aspect_subgroups'],
            proportion_method='statistical'
        )
        
        self.selected_groups['probabilistic'] = {
            'groups': selected_groups,
            'proportion': selected_proportion,
            'probabilities': probabilities,
            'metadata': prob_metadata
        }
        
        return selected_groups
    
    def _pga_weighted_selection(self):
        """Select groups using PGA-weighted method."""
        # Calculate transient acceleration groups
        factor_of_safety_grid = self.factor_of_safety_vals.reshape(self.grid.shape)
        a_transient_grid = self.critical_accelerations['no_earthquake']['transient'].reshape(self.grid.shape)
        
        a_transient_groups = np.zeros_like(factor_of_safety_grid)
        split_aspect_groups = np.unique(self.instability_regions['aspect_subgroups'])
        
        for label_number in tqdm(split_aspect_groups, desc="Processing groups"):
            if label_number > 0:
                masked_array = ~np.ma.getmask(
                    np.ma.masked_not_equal(self.instability_regions['aspect_subgroups'], label_number)
                )
                a_transient_groups[masked_array] = np.nanmedian(a_transient_grid[masked_array])
        
        pga_weighted_probabilities, proportion, pga_meta = generate_landslide_proportion_from_pga(
            self.grid,
            h_pga=self.acceleration_arrays['horizontal'],
            v_pga=self.acceleration_arrays['vertical'],
            labeled_array=self.instability_regions['aspect_subgroups'],
            weight_array=a_transient_groups
        )
        
        pga_weighted_groups, pga_weighted_group_labels = select_groups_by_proportion_weighted(
            labeled_array=self.instability_regions['aspect_subgroups'],
            probability_array=pga_weighted_probabilities,
            proportion=proportion
        )
        
        self.selected_groups['pga_weighted'] = {
            'groups': pga_weighted_groups,
            'proportion': proportion,
            'probabilities': pga_weighted_probabilities,
            'metadata': pga_meta
        }
        
        return pga_weighted_groups
    
    def calculate_region_properties(self, method: str = 'probabilistic'):
        """Calculate properties of selected regions."""
        if method not in self.selected_groups:
            raise ValueError(f"Method '{method}' not available. Run selection first.")
        
        selected_groups = self.selected_groups[method]['groups']
        
        # Calculate basic properties
        region_props, processed_groups = calculate_region_properties(
            self.grid,
            labeled_array=selected_groups,
            slopes=self.slopes_degrees,
            aspect_array=self.aspect_nodes,
            min_size=self.config['simulation']['min_region_size'],
            handle_small='merge'
        )
        
        # Remove problematic entries
        region_props = region_props[
            (region_props.area > 900) & 
            (region_props.minor_axis_length > 1)
        ].copy()
        
        self.selected_groups[method]['properties'] = region_props
        self.selected_groups[method]['processed_groups'] = processed_groups
        
        return region_props
    
    def split_wide_regions(self, method: str = 'probabilistic', kde_data=None):
        """Split wide regions based on measured landslide dimensions."""
        if method not in self.selected_groups:
            raise ValueError(f"Method '{method}' not available.")
        
        if kde_data is None:
            raise ValueError("KDE data required for region splitting")
        
        selected_groups = self.selected_groups[method]['processed_groups']
        region_props = self.selected_groups[method]['properties']
        
        new_selected_groups, split_info = split_wide_regions(
            labeled_array=selected_groups,
            region_df=region_props,
            kde_results=kde_data['kde'],
            transform_info=kde_data['transform'],
            length_col='hybrid_length',
            width_col='hybrid_width',
            label_col='label',
            width_threshold=self.config['simulation']['width_threshold']
        )
        
        # Recalculate properties for split regions
        new_region_props, new_processed_groups = calculate_region_properties(
            self.grid,
            labeled_array=new_selected_groups,
            slopes=self.slopes_degrees,
            aspect_array=self.aspect_nodes
        )
        
        # Clean up problematic entries
        new_region_props = new_region_props[
            (new_region_props.area > 900) & 
            (new_region_props.minor_axis_length > 1)
        ].copy()
        
        self.selected_groups[method + '_split'] = {
            'groups': new_selected_groups,
            'properties': new_region_props,
            'processed_groups': new_processed_groups,
            'split_info': split_info
        }
        
        return new_selected_groups, new_region_props
    
    def calculate_newmark_displacement(self, method: str = 'probabilistic'):
        """Calculate Newmark displacement for selected groups."""
        if method not in self.selected_groups:
            raise ValueError(f"Method '{method}' not available.")
        
        selected_groups = self.selected_groups[method]['processed_groups']
        a_diff_EQ = self.critical_accelerations['with_earthquake']['difference']
        
        # Ensure positive values only
        a_diff_EQ = np.maximum(a_diff_EQ, 0)
        
        # Create time array
        time_shaking = np.ones_like(selected_groups) * self.config['simulation']['time_shaking']
        
        displacement = calculate_newmark_displacement(
            self.grid,
            a_difference=a_diff_EQ,
            filtered_labeled_array=selected_groups,
            time_shaking=time_shaking
        )
        
        # Find nodes above threshold
        threshold = self.config['simulation']['displacement_threshold']
        condition_array = displacement > threshold
        node_ids = np.where(condition_array)[0]
        
        self.newmark_displacement[method] = {
            'displacement': displacement,
            'node_ids': node_ids,
            'threshold': threshold
        }
        
        return displacement, node_ids
    
    def trace_sediment_paths(self, method: str = 'probabilistic'):
        """Trace sediment transport paths."""
        if method not in self.newmark_displacement:
            raise ValueError(f"Calculate Newmark displacement for '{method}' first.")
        
        node_ids = self.newmark_displacement[method]['node_ids']
        displacement = self.newmark_displacement[method]['displacement']
        
        landslide_paths, landslide_proportions, path_details = trace_paths_landslides(
            self.grid,
            starting_nodes=node_ids,
            newmark_distances=displacement,
            check_sum=False
        )
        
        # Update soil depth
        updated_soil_depth = update_soil_depth(
            landslide_paths, landslide_proportions, self.soil_depth
        )
        
        self.landslide_paths[method] = {
            'paths': landslide_paths,
            'proportions': landslide_proportions,
            'path_details': path_details
        }
        
        self.updated_soil_depth = updated_soil_depth
        
        return landslide_paths, updated_soil_depth
    
    def calculate_final_landslide_properties(self):
        """Calculate properties of final landslides including runout."""
        if self.updated_soil_depth is None:
            raise ValueError("Run sediment transport analysis first.")
        
        # Identify final landslide areas
        landslide_areas = self.updated_soil_depth != 1.0
        landslide_areas[self.grid.boundary_nodes] = False
        landslide_areas_2d = landslide_areas.reshape(self.grid.shape)
        
        # Calculate regions
        total_labeled_array, total_num_features = calculate_regions(
            landslide_areas_2d, connect_val=8
        )
        
        # Split by aspect
        aspect_intervals = create_zones(interval=self.config['simulation']['aspect_interval'])
        aspect_nodes_array = self.aspect_nodes.reshape(self.grid.shape)
        
        total_aspect_subgroups, _, _ = split_groups_by_aspect(
            total_labeled_array, aspect_nodes_array, zones=aspect_intervals
        )
        
        # Calculate properties
        total_landslide_props = calculate_region_properties(
            self.grid,
            labeled_array=total_labeled_array,
            slopes=self.slopes_degrees,
            aspect_array=self.aspect_nodes
        )
        
        self.analysis_metadata['final_landslides'] = {
            'labeled_array': total_labeled_array,
            'aspect_subgroups': total_aspect_subgroups,
            'properties': total_landslide_props,
            'num_features': total_num_features
        }
        
        return total_landslide_props
    
    def run_complete_analysis(self, measured_data: Optional[pd.DataFrame] = None):
        """
        Run the complete landslide analysis workflow.
        
        Parameters
        ----------
        measured_data : pd.DataFrame, optional
            Measured landslide data for KDE fitting
        
        Returns
        -------
        dict
            Complete analysis results
        """
        print("Loading DEM...")
        self.load_dem()
        
        print("Setting up soil properties...")
        self.setup_soil_properties()
        
        print("Calculating slopes and aspects...")
        self.calculate_slopes_and_aspects()
        
        print("Initializing flow router...")
        self.initialize_flow_router()
        
        print("Generating acceleration grids...")
        self.generate_acceleration_grids()
        
        print("Performing stability analysis...")
        self.calculate_stability_analysis()
        
        print("Identifying failure regions...")
        self.identify_failure_regions()
        
        # Run different selection methods
        methods_config = self.config['analysis_methods']
        
        if methods_config['use_probabilistic']:
            print("Running probabilistic selection...")
            self.select_landslide_groups('probabilistic')
            self.calculate_region_properties('probabilistic')
        
        if methods_config['use_pga_weighted']:
            print("Running PGA-weighted selection...")
            self.select_landslide_groups('pga_weighted')
            self.calculate_region_properties('pga_weighted')
        
        # Split wide regions if requested and measured data available
        if methods_config['split_wide_regions'] and measured_data is not None:
            print("Fitting KDE to measured data...")
            kde_data, kde_transform = fit_bivariate_kde(
                dataframe=measured_data,
                x_col="length_m",
                y_col="width_m",
                category_col=None
            )
            
            kde_info = {'kde': kde_data, 'transform': kde_transform}
            
            for method in ['probabilistic', 'pga_weighted']:
                if method in self.selected_groups:
                    print(f"Splitting wide regions for {method}...")
                    self.split_wide_regions(method, kde_info)
        
        # Calculate Newmark displacement and trace paths
        if methods_config['trace_sediment_paths']:
            for method in self.selected_groups.keys():
                if not method.endswith('_split'):  # Use base methods for displacement
                    print(f"Calculating Newmark displacement for {method}...")
                    self.calculate_newmark_displacement(method)
                    
                    print(f"Tracing sediment paths for {method}...")
                    self.trace_sediment_paths(method)
                    break  # Only run once
        
        print("Calculating final landslide properties...")
        self.calculate_final_landslide_properties()
        
        print("Analysis complete!")
        
        return self.get_results()
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all analysis results.
        
        Returns
        -------
        dict
            Dictionary containing all analysis results and data
        """
        results = {
            'config': self.config,
            'grid': self.grid,
            'topography': {
                'elevation': self.z,
                'slopes': self.slopes_degrees,
                'aspects': self.aspect_nodes,
                'soil_depth': self.soil_depth,
                'updated_soil_depth': self.updated_soil_depth
            },
            'stability_analysis': {
                'factor_of_safety': self.factor_of_safety_vals,
                'critical_accelerations': self.critical_accelerations,
                'acceleration_arrays': self.acceleration_arrays
            },
            'instability_regions': self.instability_regions,
            'selected_groups': self.selected_groups,
            'newmark_displacement': self.newmark_displacement,
            'landslide_paths': self.landslide_paths,
            'metadata': self.analysis_metadata
        }
        
        # Add intermediate results if requested
        if self.config['output']['include_intermediate']:
            results['intermediate'] = {
                'flow_router': self.flow_router,
            }
        
        return results
    
    def save_results(self, filepath: Optional[Union[str, Path]] = None):
        """Save results to file."""
        if filepath is None:
            output_dir = self.config['output'].get('output_dir', '.')
            filepath = Path(output_dir) / 'landslide_results.pkl'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_results(), f)
        
        print(f"Results saved to {filepath}")


def load_config_from_yaml(yaml_path: Union[str, Path]) -> Dict:
    """
    Convenience function to load configuration from YAML file.
    
    Parameters
    ----------
    yaml_path : str or Path
        Path to YAML configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)