"""
Auxiliary functions for the ShallowLandslider component

author: sghoshal
"""

# Slope stability functions
from .stability import factor_of_safety, critical_transient_acceleration

from .regions import (
    calculate_regions,
    _create_zones,
    split_groups_by_aspect,
    calculate_region_properties,
)

from .selection import (
    # "probabilistic" path
    generate_landslide_probability,
    probabilistic_group_selection,
    # "pga_weighted" path
    generate_landslide_proportion_from_pga,
    select_groups_by_proportion_weighted,
)

from .split import recursive_split_wide_regions

from .utilities import (
    get_topo,
    apply_soil_depth,
    fit_bivariate_kde,
    pickle_or_not_to_pickle,
    calculate_terrain_attribute,
    generate_acceleration_grid,
)

from .displacement import calculate_newmark_displacement
