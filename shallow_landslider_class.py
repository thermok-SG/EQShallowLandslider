# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:17:46 2025

@author: sghoshal
"""

"""
Landslide Simulation Module

This module calculates:
    1. Regions of instability
    2. Filters regions by aspect
    3. Calculates Newmark displacement for each pixel
    4. Shifts sediment downslope using multi-flow approach

Functions:
    - run_simulation: Main function to run the complete landslide simulation
    - load_dem: Load digital elevation model
    - calculate_instability: Calculate static and seismic stability
    - identify_failure_regions: Identify regions of instability
    - filter_regions_by_aspect: Filter and split regions by aspect
    - calculate_displacements: Calculate Newmark displacements
    - trace_sediment_paths: Trace paths for sediment transport
    - update_topography: Update soil depth based on sediment transport

Required parameters:
    - DEM (projected)
    - PGA (both horizontal and vertical)
    - Soil parameters:
        - Cohesion (float)
        - Angle of internal friction (float)
        - Soil saturation proportion (float 0-1)
        - Soil depth (float or array of floats)

Original author: sghoshal
"""

# Core modules
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import geopandas as gpd
# import seaborn as sns
# from tqdm import tqdm

# Scientific computation
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import generate_binary_structure

# Landlab components
from landlab.io import esri_ascii
from landlab.components import PriorityFloodFlowRouter

# Import auxiliary functions
try:
    from auxiliary_functions import (
        # General functions
        get_topo, smooth_elevation_grid,
        # Functions for Newmark acceleration and displacement
        critical_transient_acceleration, calculate_newmark_displacement, factor_of_safety,
        
        # Functions for region selection and filtering
        calculate_regions, calculate_region_properties, split_groups_by_aspect, create_zones,
        
        # Group selection method 2: Select groups/proportion based on a_c
        generate_landslide_probability, probabilistic_group_selection,
        
        # Group selection method 3:
        generate_landslide_proportion_from_pga, select_groups_by_proportion_weighted,
        
        # Functions for tracing paths and moving sediment
        trace_paths_landslides, generate_acceleration_grid, update_soil_depth
    )
except ImportError:
    print("Warning: Could not import auxiliary functions. Make sure they're in your PYTHONPATH.")


class ShallowLandslideSimulator:
    """
    Main class for landslide simulation.
    """
    
    def __init__(self, config=None, grid=None, process_grid=False):
        """
        Initialize landslide simulation with configuration.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary with simulation parameters.
            If None, default parameters will be used.
        grid : landlab.grid.RasterModelGrid, optional
            Pre-loaded Landlab grid with elevation data.
            If provided, this will bypass DEM loading in the load_dem method.
        """
        # Set default config if not provided
        if config is None:
            self.config = {
                'dem_info': {
                    'dem_type': "SRTMGL1",
                    'north': 28.29,
                    'east': 85.20,
                    'south': 28.18,
                    'west': 85.04,
                    'buffer': 0.01,
                    'smooth_num': 4
                },
                'soil_params': {
                    'angle_int_frict': np.radians(30),
                    'cohesion_eff': 15e3,  # Pa
                    'submerged_soil_proportion': 0.5,
                    'soil_depth': 1.0  # m
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
                },
                'output': {
                    'save_plots': False,
                    'output_dir': None,
                    'plot_intermediate': False
                }
            }
        else:
            self.config = config
        
        # Store the pre-loaded grid if provided
        self.grid = grid
        if self.grid is not None:
            # Get the elevation data from the grid
            self.z = self.grid.at_node["topographic__elevation"]
        else:
            self.grid = None
            self.z = None
            
        # Initialize variables
        self.slopes = None
        self.slopes_degrees = None
        self.factor_of_safety_vals = None
        self.a_transient_zero = None
        self.a_transient_EQ = None
        self.a_sliding_EQ = None
        self.a_diff_EQ = None
        self.sliding_locations = None
        self.sliding_locations_bool = None
        self.sliding_array_bool = None
        self.labeled_array = None
        self.labeled_array_filled = None
        self.aspect_nodes = None
        self.aspect_subgroups = None
        self.acceleration_horizontal_array = None
        self.acceleration_vertical_array = None
        self.pga_weighted_groups = None
        self.selected_groups = None
        self.newmark_displacement_select = None
        self.node_ids = None
        self.landslide_paths = None
        self.updated_soil_depth = None
        
        # Results storage
        self.results = {}
        
        # If grid is provided, process it
        if self.grid is not None and process_grid:
            self._process_grid()
    
    def _process_grid(self):
        """
        Process a loaded grid by setting up soil depth, calculating slopes, and running flow router.
        """
        # Add soil depth field if it doesn't exist
        if "soil__depth" not in self.grid.at_node:
            soil_depth = self.grid.add_zeros("node", "soil__depth", clobber=True)
            soil_depth_value = self.config['soil_params']['soil_depth']
            
            # Set initial soil depth at core nodes
            self.grid.at_node["soil__depth"][self.grid.core_nodes] = soil_depth_value
            
        # Add bedrock elevation field if it doesn't exist
        if "bedrock__elevation" not in self.grid.at_node:
            self.grid.add_zeros("bedrock__elevation", at="node", clobber=True)
            
            # Set bedrock elevation
            self.grid.at_node["bedrock__elevation"][:] = self.grid.at_node["topographic__elevation"] - self.grid.at_node["soil__depth"]
        
        # Calculate slopes
        self.slopes = self.grid.calc_slope_at_node(elevs='topographic__elevation')
        self.slopes_degrees = np.degrees(self.slopes)
        
        # Initialize and run flow router
        pf = PriorityFloodFlowRouter(
            self.grid, 
            flow_metric='D8', 
            separate_hill_flow=True,
            depression_handler="fill", 
            update_hill_depressions=True,
            accumulate_flow=True
        )
        pf.run_one_step()
        
        # Store results
        self.results['dem'] = {
            'grid': self.grid,
            'z': self.z,
            'slopes': self.slopes,
            'slopes_degrees': self.slopes_degrees
        }
        
    def load_dem(self, file_path=None):
        """
        Load digital elevation model either from file or using get_topo.
        If a grid was provided during initialization, this method will return that grid.
        
        Parameters:
        -----------
        file_path : str, optional
            Path to DEM file. If None, will use get_topo with dem_info config.
        
        Returns:
        --------
        grid : landlab.grid.RasterModelGrid
            Landlab grid with elevation data
        """
        # If grid was already provided during initialization, just process and return it
        if self.grid is not None:
            return self.grid
            
        if file_path:
            # Load DEM from file
            with open(file_path) as fp:
                self.grid = esri_ascii.load(fp, name="topographic__elevation", at="node")
            self.z = self.grid.at_node["topographic__elevation"]
        else:
            # Generate DEM using get_topo
            dem_info = self.config['dem_info']
            self.grid, self.z = get_topo(
                dem_type=dem_info['dem_type'],
                north=dem_info['north'], south=dem_info['south'],
                east=dem_info['east'], west=dem_info['west'],
                buffer=dem_info['buffer']
            )
        
            # Smooth the generated DEM
            smoothed_elev = smooth_elevation_grid(
                self.grid, 
                method='gaussian', 
                smooth_num=self.config['dem_info']['smooth_num']
            )
            self.grid.at_node["topographic__elevation"] = smoothed_elev
        
        # Open all boundary nodes
        self.grid.set_closed_boundaries_at_grid_edges(
            right_is_closed=False, top_is_closed=False,
            left_is_closed=False, bottom_is_closed=False
        )
        
        # Process the grid (add soil depth, calculate slopes, etc.)
        self._process_grid()
        
        return self.grid
    
    def calculate_instability(self):
        """
        Calculate static and seismic stability.
        
        Returns:
        --------
        dict
            Dictionary containing stability calculation results
        """
        # Extract soil parameters
        soil_params = self.config['soil_params']
        angle_int_frict = soil_params['angle_int_frict']
        cohesion_eff = soil_params['cohesion_eff']
        submerged_soil_proportion = soil_params['submerged_soil_proportion']
        
        # Create PGA arrays
        pga_config = self.config['pga']
        self.acceleration_horizontal_array, self.acceleration_vertical_array = generate_acceleration_grid(
            self.grid,
            max_horizontal=pga_config['horizontal'],
            max_vertical=pga_config['vertical'],
            distribution=pga_config['distribution']
        )
        
        # Calculate factor of safety
        self.factor_of_safety_vals = factor_of_safety(
            self.grid, 
            cohesion_eff, 
            angle_int_frict
        )
        
        # Calculate critical acceleration with no earthquake
        self.a_transient_zero, a_sliding_zero, a_diff_zero = critical_transient_acceleration(
            self.grid, 
            cohesion_eff, 
            angle_int_frict,
            submerged_soil_proportion=submerged_soil_proportion, 
            a_h=0, 
            a_v=0
        )
        
        # Calculate critical acceleration due to earthquake
        g = 9.81  # Acceleration due to gravity (m/s^2)
        self.a_transient_EQ, self.a_sliding_EQ, self.a_diff_EQ = critical_transient_acceleration(
            self.grid, 
            cohesion_eff, 
            angle_int_frict, 
            submerged_soil_proportion=submerged_soil_proportion,
            a_h=self.acceleration_horizontal_array*g, 
            a_v=self.acceleration_vertical_array*g
        )
        
        # Areas of instability
        self.sliding_locations_bool = self.a_sliding_EQ > self.a_transient_EQ
        
        # Create array of pseudo-bool values for sliding locations
        self.sliding_locations = np.double(self.sliding_locations_bool)
        
        # Process boundaries
        self.sliding_locations_bool[self.grid.boundary_nodes] = False
        self.sliding_array_bool = self.sliding_locations_bool.reshape(self.grid.shape)
        
        # Store results
        self.results['instability'] = {
            'factor_of_safety': self.factor_of_safety_vals,
            'a_transient_zero': self.a_transient_zero,
            'pga_h_array': self.acceleration_horizontal_array,
            'pga_v_array': self.acceleration_vertical_array,
            'a_transient_EQ': self.a_transient_EQ,
            'a_sliding_EQ': self.a_sliding_EQ,
            'a_diff_EQ': self.a_diff_EQ,
            'sliding_locations': self.sliding_locations,
            'sliding_locations_bool': self.sliding_locations_bool
        }
        
        return self.results['instability']
    
    def identify_failure_regions(self):
        """
        Identify regions of instability.
        
        Returns:
        --------
        dict
            Dictionary containing region identification results
        """
        # Create binary structure for grouping
        struct = generate_binary_structure(2, 2)
        
        # Identify separate regions from unstable zones
        self.labeled_array, num_features = calculate_regions(
            self.sliding_array_bool, 
            connect_val=8
        )
        
        labeled_groups = np.unique(self.labeled_array)
        
        # Fill-in holes for each labeled group
        self.labeled_array_filled = np.zeros_like(self.labeled_array)
        for label_number in labeled_groups:
            masked_array = ~np.ma.getmask(
                np.ma.masked_not_equal(self.labeled_array, label_number)
            )
            if label_number > 0:
                self.labeled_array_filled[binary_fill_holes(
                    masked_array, structure=struct
                )] = label_number
        
        # Store results
        self.results['regions'] = {
            'labeled_array': self.labeled_array,
            'labeled_array_filled': self.labeled_array_filled,
            'num_features': num_features
        }
        
        return self.results['regions']
    
    def filter_regions_by_aspect(self):
        """
        Filter and split regions by aspect.
        
        Returns:
        --------
        dict
            Dictionary containing aspect filtering results
        """
        # Calculate topographic aspect
        self.aspect_nodes = np.array(self.grid.calc_aspect_at_node(
            elevs='topographic__elevation', 
            unit="degrees",
            ignore_closed_nodes=True
        ))
        
        self.aspect_nodes[self.grid.boundary_nodes] = np.nan
        aspect_nodes_array = self.aspect_nodes.reshape(self.grid.shape)
        
        # Create dictionary of aspect zones
        aspect_intervals = create_zones(interval=self.config['simulation']['aspect_interval'])
        
        # Split groups by aspect
        self.aspect_subgroups, aspect_zones, info = split_groups_by_aspect(
            self.labeled_array_filled,
            aspect_nodes_array,
            zones=aspect_intervals
        )
        
        split_aspect_groups = np.unique(self.aspect_subgroups)
        split_aspect_group_num = len(split_aspect_groups)
        
        # Assign factors of safety and critical acceleration to each labeled group
        factor_of_safety_grid = self.factor_of_safety_vals.reshape(self.grid.shape)
        a_transient_grid = self.a_transient_zero.reshape(self.grid.shape)
        
        factor_of_safety_groups = np.zeros_like(factor_of_safety_grid)
        a_transient_groups = np.zeros_like(factor_of_safety_grid)
        
        for label_number in split_aspect_groups:
            masked_array = ~np.ma.getmask(
                np.ma.masked_not_equal(self.aspect_subgroups, label_number)
            )
            if label_number > 0:
                factor_of_safety_groups[masked_array] = np.nanmedian(factor_of_safety_grid[masked_array])
                a_transient_groups[masked_array] = np.nanmedian(a_transient_grid[masked_array])
            else:
                factor_of_safety_groups[masked_array] = 0
                a_transient_groups[masked_array] = 0
        
        # Calculate region properties
        subgroup_props = calculate_region_properties(
            self.grid, 
            self.aspect_subgroups, 
            slopes=self.slopes_degrees
        )
        
        # Store results
        self.results['aspect_filtering'] = {
            'aspect_nodes': self.aspect_nodes,
            'aspect_subgroups': self.aspect_subgroups,
            'aspect_zones': aspect_zones,
            'split_aspect_groups': split_aspect_groups,
            'split_aspect_group_num': split_aspect_group_num,
            'factor_of_safety_groups': factor_of_safety_groups,
            'a_transient_groups': a_transient_groups,
            'subgroup_props': subgroup_props
        }
        
        return self.results['aspect_filtering']
    
    def select_potential_landslides(self, method='pga_weighted'):
        """
        Select potential landslide groups using various methods.
        
        Parameters:
        -----------
        method : str
            Method for selecting landslide groups:
            - 'pga_weighted': Selection based on PGA-weighted probabilities
            - 'probabilistic': Selection based on critical acceleration
        
        Returns:
        --------
        dict
            Dictionary containing landslide selection results
        """
        if method == 'pga_weighted':
            # Method 2: Select groups/proportion from PGA
            pga_weighted_probabilities, proportion, pga_meta = generate_landslide_proportion_from_pga(
                self.grid,
                h_pga=self.acceleration_horizontal_array,
                v_pga=self.acceleration_vertical_array,
                labeled_array=self.aspect_subgroups,
                weight_array=self.results['aspect_filtering']['a_transient_groups']
            )
            
            self.pga_weighted_groups, pga_weighted_group_labels = select_groups_by_proportion_weighted(
                labeled_array=self.aspect_subgroups,
                probability_array=pga_weighted_probabilities,
                proportion=proportion
            )
            
            # Calculate properties
            pga_weighted_group_props = calculate_region_properties(
                self.grid, 
                labeled_array=self.pga_weighted_groups,
                slopes=self.slopes_degrees
            )
            
            # Store results
            self.results['selected_landslides'] = {
                'method': 'pga_weighted',
                'probabilities': pga_weighted_probabilities,
                'proportion': proportion,
                'meta': pga_meta,
                'groups': self.pga_weighted_groups,
                'group_labels': pga_weighted_group_labels,
                'group_props': pga_weighted_group_props
            }
            
            # Set selected groups for further processing
            self.selected_groups = self.pga_weighted_groups
            
        elif method == 'probabilistic':
            # Method 3: Select groups/proportion based on critical acceleration
            probabilities, prob_metadata = generate_landslide_probability(
                self.grid,
                h_pga_array=self.acceleration_horizontal_array,
                v_pga_array=self.acceleration_vertical_array,
                labeled_array=self.aspect_subgroups,
                slope_array=self.slopes_degrees,
                soil_array=None,
                geological_factor_array=None,
                critical_acceleration_array=self.a_transient_zero,
                default_critical_acceleration=0.2,
                random_seed=None,
                normalise_final_probs=True
            )
            
            self.selected_groups, selected_proportion = probabilistic_group_selection(
                probability_array=probabilities,
                labeled_array=self.aspect_subgroups,
                proportion_method='statistical'
            )
            
            # Calculate areas and other parameters for the selected groups
            selected_group_props = calculate_region_properties(
                self.grid, 
                labeled_array=self.selected_groups, 
                slopes=self.slopes_degrees
            )
            
            # Store results
            self.results['selected_landslides'] = {
                'method': 'probabilistic',
                'probabilities': probabilities,
                'proportion': selected_proportion,
                'meta': prob_metadata,
                'groups': self.selected_groups,
                'group_props': selected_group_props
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.results['selected_landslides']
    
    def calculate_displacements(self):
        """
        Calculate Newmark displacements for selected landslide groups.
        
        Returns:
        --------
        dict
            Dictionary containing displacement results
        """
        time_shaking = np.ones_like(self.aspect_subgroups) * self.config['simulation']['time_shaking']
        
        # Ensure a_diff_EQ is non-negative
        a_diff_EQ_copy = self.a_diff_EQ.copy()
        a_diff_EQ_copy[a_diff_EQ_copy < 0] = 0
        
        # Calculate Newmark displacement for all unstable pixels
        newmark_displacement_all = calculate_newmark_displacement(
            self.grid,
            a_difference=a_diff_EQ_copy,
            filtered_labeled_array=self.aspect_subgroups,
            time_shaking=time_shaking
        )
        
        # Calculate Newmark displacement for selected groups
        self.newmark_displacement_select = calculate_newmark_displacement(
            self.grid,
            a_difference=a_diff_EQ_copy,
            filtered_labeled_array=self.selected_groups,
            time_shaking=time_shaking
        )
        
        # Find nodes with high displacement
        displacement_threshold = self.config['simulation']['displacement_threshold']
        
        condition_array = np.zeros(self.grid.number_of_nodes, dtype=bool)
        condition_array[self.newmark_displacement_select > displacement_threshold] = True
        
        self.node_ids = np.where(condition_array)[0]
        
        # Store results
        self.results['displacements'] = {
            'newmark_displacement_all': newmark_displacement_all,
            'newmark_displacement_select': self.newmark_displacement_select,
            'high_displacement_nodes': self.node_ids,
            'displacement_threshold': displacement_threshold
        }
        
        return self.results['displacements']
    
    def trace_sediment_paths(self):
        """
        Trace paths for sediment transport and calculate changes in soil depth.
        
        Returns:
        --------
        dict
            Dictionary containing sediment transport results
        """
        # Calculate paths and update soil depth
        self.landslide_paths, landslide_proportions, path_details = trace_paths_landslides(
            self.grid,
            starting_nodes=self.node_ids,
            newmark_distances=self.newmark_displacement_select,
            check_sum=False
        )
        
        # Calculate changes in soil depth
        self.updated_soil_depth = update_soil_depth(
            self.landslide_paths,
            landslide_proportions,
            self.grid.at_node['soil__depth']
        )
        
        # Calculate mass balance
        prev_soil_depth = np.sum(self.grid.at_node['soil__depth'])
        new_soil_depth = np.sum(self.updated_soil_depth)
        mass_balance = new_soil_depth - prev_soil_depth
        
        delta_soil_depth = self.updated_soil_depth - self.grid.at_node['soil__depth']
        
        # Store results
        self.results['sediment_transport'] = {
            'landslide_paths': self.landslide_paths,
            'landslide_proportions': landslide_proportions,
            'path_details': path_details,
            'updated_soil_depth': self.updated_soil_depth,
            'mass_balance': mass_balance,
            'soil_depth_change': delta_soil_depth
        }
        
        return self.results['sediment_transport']
    
    def run_one_step(self, landslide_selection_method='probabilistic'):
        """
        Run a single timestep of the landslide simulation, executing each step in sequence.
        
        Parameters:
        -----------
        landslide_selection_method : str, optional
            Method for selecting landslide groups:
            - 'pga_weighted': Selection based on PGA-weighted probabilities
            - 'probabilistic': Selection based on critical acceleration
        
        Returns:
        --------
        dict
            Dictionary containing output data from all simulation steps for plotting and analysis
        """
        # Step 3: Calculate instability
        self.calculate_instability()
        
        # Step 4: Identify failure regions
        self.identify_failure_regions()
        
        # Step 5: Filter regions by aspect
        self.filter_regions_by_aspect()
        
        # Step 6: Select potential landslides
        self.select_potential_landslides(method=landslide_selection_method)
        
        # Step 7: Calculate displacements
        self.calculate_displacements()
        
        # Step 8: Trace sediment paths and update topography
        self.trace_sediment_paths()
        
        # Update the grid with new soil depth
        self.grid.at_node['soil__depth'] = self.updated_soil_depth
        
        # Update topographic elevation based on new soil depth
        self.grid.at_node['topographic__elevation'] = (
            self.grid.at_node['bedrock__elevation'] + self.grid.at_node['soil__depth']
        )
        
        # Recalculate slopes with updated topography
        self.slopes = self.grid.calc_slope_at_node(elevs='topographic__elevation')
        self.slopes_degrees = np.degrees(self.slopes)
        
        # Compile plotting data for return
        plot_data = {
            # Topography data
            'slopes': self.slopes_degrees,
            
            # Instability factors
            'factor_of_safety': self.factor_of_safety_vals,
            'critical_acceleration': self.a_transient_EQ,
            'sliding_areas': self.sliding_array_bool,
            
            # Landslide regions
            'landslide_regions': self.labeled_array_filled,
            'aspect_subgroups': self.aspect_subgroups,
            'selected_landslides': self.selected_groups,
            'landslide_proportion': self.results['selected_landslides']['selected_proportion'],
            
            # Displacements and sediment transport
            'displacements': self.newmark_displacement_select,
            'soil_depth_change': self.results['sediment_transport']['soil_depth_change'],
        }
        
        return self.grid, plot_data