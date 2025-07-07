"""
Auxiliary functions for the ShallowLandslider component

author: sghoshal
"""

from .io import get_config

from .terrain import (
    get_topo, smooth_elevation_grid,
    apply_soil_depth, 
    )

from .newmark import (
    factor_of_safety,
    critical_transient_acceleration,
    calculate_newmark_displacement
    )

from .regions import (
    calculate_regions,
    create_zones, split_groups_by_aspect,
    calculate_region_properties
    )

from .selection import (
    # Group selection method 2: Select groups/proportion based on a_c
    generate_landslide_probability, probabilistic_group_selection,
    
    # Group selection method 3:
    generate_landslide_proportion_from_pga, select_groups_by_proportion_weighted
    )

from .simulation import (
    generate_acceleration_grid,
    trace_paths_landslides,
    update_soil_depth
    )

from .stats import (
    # Splits regions that are too wide compared to their length
    split_wide_regions,
    recursive_split_wide_regions, analyze_split_results,
    
    # Statistical fits
    fit_bivariate_kde, plot_bivariate_kde,
    conditional_sample, plot_conditional_samples
    )

from .topographic_functions import (
    # Excess topography
    calculate_excess_topography
)