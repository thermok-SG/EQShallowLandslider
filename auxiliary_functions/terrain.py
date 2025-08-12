"""
auxiliary_functions/terrain.py

Functions to set up the grid
"""
# %% Import required packages
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from landlab import RasterModelGrid
from landlab.io import esri_ascii

from bmi_topography import Topography

# %% Getting topography from OpenTopography
def get_topo(buffer: float, north=28.25, south=28.23, east=85.18, west=85.15, dem_type="NASADEM"):
    """
    Downloads DEM from OpenTopo and generates a landlab RasterModelGrid.

    Parameters
    ----------
    buffer : float
        Additional space around the DEM to remove potential edge effects.
        Values in decimal degrees.
    north : float, optional
        Northern extent of DEM in decimal degrees. The default is 28.25.
    south : float, optional
        Southern extent of DEM in decimal degrees. The default is 28.23.
    east : float, optional
        Eastern extent of DEM in decimal degrees. The default is 85.18.
    west : float, optional
        Western extent of DEM in decimal degrees. The default is 85.15.
    dem_type : string, optional
        Type of DEM to download from OpenTopo. The default is "NASADEM".
        Available DEM types:
            1. SRTMGL3: Default, 90 m
            2. SRTMGL1: 30 m
            3. AW3D30: ALOS World 3D, 30 m
            4. NASADEM: NASADEM Global DEM, 30 m
            5. COP30: Copernicus Global DSM 30 m
            6. COP90: Copernicus Global DSM 90 m

    Returns
    -------
    grid : RasterModelGrid
        Elevation grid.
    z_geog : ndarray
        Array of elevation values.

    """

    params = Topography.DEFAULT.copy()
    params["south"] = south - buffer
    params["north"] = north + buffer
    params["west"] = west - buffer
    params["east"] = east + buffer
    params["dem_type"] = dem_type
    params["output_format"] = "AAIGrid"
    params["cache_dir"] = Path.cwd()
    params["api_key"] = "f08b2664772eb044626d5cb114924de1"
    dem = Topography(**params)
    name = dem.fetch()
    props = dem.load()
    
    with open(name) as fp:
        grid_geog = esri_ascii.load(fp, name="topographic__elevation", at="node")
    
    z_geog = grid_geog.at_node["topographic__elevation"]
    
    match dem_type:
        case "SRTMGL3" | "COP90":
            grid_spacing = 90
        case "SRTMGL1" | "AW3D30" | "NASADEM" | "COP30":
            grid_spacing = 30
    
    grid = RasterModelGrid(
        (grid_geog.number_of_node_rows, grid_geog.number_of_node_columns),
        xy_spacing=grid_spacing,
        xy_axis_units='m',
    )
    grid.add_field("topographic__elevation", z_geog, at="node")

    print(params)
    print(props)
    
    return grid, z_geog
# %% Smoothen landlab grid
def smooth_elevation_grid(grid, method='mean', smooth_num=1):
    """
    Apply a 3x3 moving window smoothing to a landlab grid's elevation data.
    
    Parameters:
    -----------
    grid : landlab.grid
        The landlab grid containing elevation data
    method : str, optional
        The smoothing method to use: 'mean' or 'gaussian' (default: 'mean')
        
    Returns:
    --------
    numpy.ndarray
        The smoothed elevation array
    """
    if method == 'mean':
        from scipy.ndimage import uniform_filter
    elif method == 'gaussian':
        # For Gaussian smoothing
        from scipy.ndimage import gaussian_filter
    else:
        raise ValueError("Method must be 'mean' or 'gaussian'")
    
    # Get the original elevation data
    elevation = grid.at_node['topographic__elevation'].copy()
    
    # Reshape to 2D for smoothing (assuming a raster grid)
    elevation_2d = elevation.reshape(grid.shape)
    
    smooth_round = 0
    while smooth_round < smooth_num:
    
        if method == 'mean':
            # Apply a 3x3 mean filter
            smoothed_elevation = uniform_filter(elevation_2d, size=3, mode='reflect')
        elif method == 'gaussian':
            # For Gaussian smoothing
            smoothed_elevation = gaussian_filter(elevation_2d, sigma=1, mode='reflect')
        smooth_round += 1
    
    # Reshape back to 1D for use with landlab
    smoothed_elevation_1d = smoothed_elevation.flatten()
    
    return smoothed_elevation_1d

# %% Generate soil depth array
def apply_soil_depth(grid,
                    elevation_field='topographic__elevation', 
                    soil_field='soil__depth',
                    max_soil_depth=1.0,
                    distribution='uniform',
                    relationship='linear',
                    decay_rate=5.0,
                    exponent=1.0,
                    plot=False
                    ):
    """
    Apply soil depth to core nodes based on elevation or uniform distribution.
    
    Parameters:
    -----------
    grid : Landlab grid object
        The landlab grid (RasterModelGrid, HexModelGrid, etc.)
    elevation_field : str, default 'topographic__elevation'
        Name of the field containing elevation data
    soil_field : str, default 'soil__depth'
        Name of the field to store soil depth data
    max_soil_depth : float, default 1.0
        Maximum soil depth in meters. For elevation-based: applied to minimum elevation.
        For uniform: applied to all core nodes.
    distribution : str, default 'uniform'
        When 'uniform' : 
        - All core nodes get the same soil depth (max_soil_depth)
        - Boundary nodes get zero soil depthapplies the max_soil_depth uniformly across entire grid
        When 'elevation' :
        - Soil depth varies inversely with elevation according to specified relationships
    relationship : str, default 'linear'
        Type of relationship between elevation and soil depth (only used when distribution='elevation'):
        - 'linear': Linear inverse relationship (original behavior)
        - 'exponential': Exponential decay with elevation
        - 'power': Power law relationship using exponent parameter
        - 'logarithmic': Logarithmic relationship (gentle at low elevations, steep at high)
        - 'sigmoid': S-shaped curve (smooth transition)
    decay_rate : float, default 5.0
        Rate of decay of exponential relationship
        Higher values mean more soil concentrated at lower elevations
    exponent : float, default 1.0
        Exponent for power law relationship. Only used when relationship='power'.
        - exponent > 1: More soil at low elevations (concave down)
        - exponent < 1: More gradual decrease (concave up)
        - exponent = 1: Linear relationship
    plot : bool, default False
        If True, creates a scatter plot showing soil depth vs elevation relationship.
    
    Returns:
    --------
    soil_depth : ndarray
        1D array of soil depth values, associated with the grid 'soil__depth' field
        
    """
    # Get elevation data and core nodes
    elevation = grid.at_node[elevation_field]
    core_nodes = grid.core_nodes
    
    # Initialize soil depth field on the grid (zeros everywhere)
    soil_depth = grid.add_zeros("node", soil_field, clobber=True)
    
    if distribution == 'uniform':
        # Apply uniform soil depth to core nodes only
        max_soil_depth = 1.0 # meter
        soil_depth[core_nodes] = max_soil_depth
        print("Uniform soil depth applied to grid:")
        print(f"  Soil depth: {max_soil_depth:.2f} m (uniform)")
        print(f"  Core nodes processed: {len(core_nodes)}")
        
    elif distribution == 'elevation':
        # Apply elevation-based soil depth
        core_elevations = elevation[core_nodes]
        
        # Calculate min and max elevation from core nodes
        min_elevation = np.min(core_elevations)
        max_elevation = np.max(core_elevations)
        
        # Check if there's any elevation variation
        elevation_range = max_elevation - min_elevation
        if elevation_range == 0:
            # If all elevations are the same, assign uniform soil depth
            soil_depth[core_nodes] = max_soil_depth / 2.0
            print("Warning: All core nodes have the same elevation. Assigning uniform soil depth.")
        else:
            # Calculate normalized elevation (0 = min elevation, 1 = max elevation)
            normalized_elevation = (elevation - min_elevation) / elevation_range
            
            if relationship == 'linear':
                # Original linear inverse relationship
                temp_soil_depth = max_soil_depth * (1.0 - normalized_elevation)
            
            elif relationship == 'exponential':
                # Normalized exponential decay: more soil concentrated at low elevations
                # Ensures 0 m soil at max elevation
                top_term = np.exp(-decay_rate)
                temp_soil_depth = max_soil_depth * (np.exp(-decay_rate * normalized_elevation) - top_term) / (1 - top_term)
            
            elif relationship == 'power':
                # Power law relationship using exponent parameter
                temp_soil_depth = max_soil_depth * (1.0 - normalized_elevation) ** exponent
            
            # Better to just use exponential
            # elif relationship == 'logarithmic':
            #     # Logarithmic relationship: gentle decrease at low elevations, steep at high
            #     epsilon = 0.001
            #     temp_soil_depth = max_soil_depth * (1.0 - np.log(normalized_elevation + epsilon) / np.log(1.0 + epsilon))
            
            elif relationship == 'sigmoid':
                # Sigmoid relationship: S-shaped curve
                # Maps normalized elevation through sigmoid function
                sigmoid_input = 6.0 * (normalized_elevation - 0.5)  # Scale to reasonable range
                sigmoid_output = 1.0 / (1.0 + np.exp(sigmoid_input))
                temp_soil_depth = max_soil_depth * sigmoid_output
            
            else:
                raise ValueError(f"Unknown relationship type: {relationship}. Must be one of: 'linear', 'exponential', 'power', 'logarithmic', 'sigmoid'")
            
            # Ensure non-negative values
            temp_soil_depth = np.maximum(temp_soil_depth, 0.0)
            
            # Copy values to the grid field
            soil_depth[:] = temp_soil_depth
            
            # Set boundary nodes to zero
            boundary_nodes = np.setdiff1d(np.arange(grid.number_of_nodes), core_nodes)
            soil_depth[boundary_nodes] = 0.0
        
        # Print summary statistics
        print("Elevation-based soil depth applied to grid:")
        print(f"  Elevation range: {min_elevation:.2f} to {max_elevation:.2f} m")
        print(f"  Soil depth range: 0.00 to {max_soil_depth:.2f} m")
        print(f"  Core nodes processed: {len(core_nodes)}")
    
    else:
        raise ValueError("Soil distribution can only be 'uniform' or 'elevation'")
    
    # Create plot if requested
    if plot:
        create_soil_depth_plot(grid, elevation_field, soil_field,
                                distribution, relationship)
    
    return soil_depth

# %%% Helper functions to create plot of soil depth distribution
def create_soil_depth_plot(grid,
                            elevation_field,
                            soil_field,
                            distribution,
                            relationship
                            ):
    """
    Create a scatter plot showing the relationship between elevation and soil depth.
    
    Parameters:
    -----------
    grid : Landlab grid object
        The landlab grid with elevation and soil depth fields
    elevation_field : str
        Name of the elevation field
    soil_field : str
        Name of the soil depth field
    distribution : str
        How soil depth was applied across the grid
    relationship : str
        Mathematical relationship between elevation and soil depth
    """
    # Get data for core nodes only
    elevation = grid.at_node[elevation_field]
    soil_depth = grid.at_node[soil_field]
    core_nodes = grid.core_nodes
    
    core_elevation = elevation[core_nodes]
    core_soil_depth = soil_depth[core_nodes]
    
    # Create the plot
    plt.figure(layout='constrained')
    
    if distribution == 'uniform':
        # For uniform case, show horizontal line
        plt.scatter(core_elevation, core_soil_depth, alpha=0.6, s=30, color='blue',
                    label=f'Core nodes (n={len(core_nodes)})')
        plt.axhline(y=core_soil_depth[0], color='red', linestyle='--', alpha=0.7,
                    label=f'Uniform depth = {core_soil_depth[0]:.2f} m')
        plt.title('Uniform Soil Depth Distribution')
    else:
        # For elevation-based case, show scatter with trend line
        plt.scatter(core_elevation, core_soil_depth, alpha=0.6, s=30, color='blue',
                    label=f'Core nodes (n={len(core_nodes)})')
        plt.title(f'{relationship.capitalize()}-based soil depth distribution')
        
        # Add theoretical trend line
        elev_range = np.linspace(core_elevation.min(), core_elevation.max(), 100)
        max_depth = core_soil_depth.max()
        min_elev = core_elevation.min()
        max_elev = core_elevation.max()
        
        if max_elev > min_elev:  # Avoid division by zero
            normalized_elev = (elev_range - min_elev) / (max_elev - min_elev)
            theoretical_depth = max_depth * (1.0 - normalized_elev)
            plt.plot(elev_range, theoretical_depth, 'r--', alpha=0.7, 
                    label='Theoretical relationship')
        
        plt.title('Elevation-Based Soil Depth Distribution')
    
    plt.xlabel('Elevation (m)')
    plt.ylabel('Soil Depth (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics text box
    stats_text = 'Statistics:\n'
    stats_text += f'Elevation: {core_elevation.min():.2f} - {core_elevation.max():.2f} m\n'
    stats_text += f'Soil depth: {core_soil_depth.min():.2f} - {core_soil_depth.max():.2f} m\n'
    stats_text += f'Mean soil depth: {core_soil_depth.mean():.2f} m'
    
    plt.text(0.02, 0.22, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# %%% Helper functions to compare soil depth distributions
def compare_soil_relationships(grid, elevation_field='topographic__elevation', max_soil_depth=1.0, 
                                plot=True, plot_type='combined'):
    """
    Compare different soil depth relationships and optionally create visualization plots.
    
    Parameters:
    -----------
    grid : Landlab grid object
        The landlab grid
    elevation_field : str, default 'topographic__elevation'
        Name of the elevation field
    max_soil_depth : float, default 1.0
        Maximum soil depth in meters
    plot : bool, default True
        Whether to create plots
    plot_type : str, default 'combined'
        Type of plot to create:
        - 'combined': Single plot with all relationships
        - 'individual': Separate plot for each relationship
        - 'grid': Subplot grid showing all relationships
    
    Returns:
    --------
    dict : Dictionary containing soil depth arrays for each relationship
    """
    
    # Define relationships to test
    relationships = [
        ('linear', 1.0),
        ('exponential', 1.0),
        ('power', 0.5),
        ('power', 1.0), 
        ('power', 2.0),
        ('logarithmic', 1.0),
        ('sigmoid', 1.0)
    ]
    
    # Get elevation data for plotting
    elevation = grid.at_node[elevation_field]
    core_nodes = grid.core_nodes
    core_elevation = elevation[core_nodes]
    
    # Storage for results
    results = {}
    
    print("Comparing soil depth relationships:")
    print("=" * 60)
    
    # Calculate soil depths for each relationship
    for rel, exp in relationships:
        # Create field name
        if rel == 'power':
            field_name = f'soil__{rel}_exp{exp}'
            display_name = f'{rel} (exp={exp})'
        else:
            field_name = f'soil__{rel}'
            display_name = rel
            
        # Apply soil depth
        soil_depth = apply_soil_depth(grid, 
                                    elevation_field=elevation_field,
                                    soil_field=field_name,
                                    max_soil_depth=max_soil_depth,
                                    distribution='elevation',
                                    relationship=rel,
                                    exponent=exp,
                                    plot=False)  # Don't plot individual ones
        
        results[display_name] = soil_depth[core_nodes]
        print()
    
    # Create plots if requested
    if plot:
        if plot_type == 'combined':
            _create_combined_plot(core_elevation, results, elevation_field)
        elif plot_type == 'individual':
            _create_individual_plots(grid, elevation_field, relationships, max_soil_depth)
        elif plot_type == 'grid':
            _create_grid_plot(core_elevation, results, elevation_field)
        else:
            raise ValueError("plot_type must be 'combined', 'individual', or 'grid'")
    
    return results

def _create_combined_plot(core_elevation, results, elevation_field):
    """Create a single plot comparing all relationships."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(12, 8), layout='constrained')
    
    # Color palette for different relationships
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (name, soil_depths) in enumerate(results.items()):
        plt.scatter(core_elevation, soil_depths, alpha=0.7, s=20, 
                    color=colors[i], label=name)
    
    plt.xlabel('Elevation (m)')
    plt.ylabel('Soil Depth (m)')
    plt.title('Comparison of Soil Depth Relationships')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add summary statistics
    stats_text = f'Elevation range: {core_elevation.min():.2f} - {core_elevation.max():.2f} m\n'
    stats_text += f'Core nodes: {len(core_elevation)}'
    
    plt.text(0.02, 0.15, stats_text, transform=plt.gca().transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()

def _create_grid_plot(core_elevation, results, elevation_field):
    """Create a subplot grid showing all relationships."""

    n_plots = len(results)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), layout='constrained')
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (name, soil_depths) in enumerate(results.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        ax.scatter(core_elevation, soil_depths, alpha=0.7, s=20, color='blue')
        ax.set_xlabel('Elevation (m)')
        ax.set_ylabel('Soil Depth (m)')
        ax.set_title(f'{name.capitalize()} Relationship')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Range: {soil_depths.min():.2f} - {soil_depths.max():.2f} m\n'
        stats_text += f'Mean: {soil_depths.mean():.2f} m'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.show()


def _create_individual_plots(grid, elevation_field, relationships, max_soil_depth):
    """Create individual plots for each relationship."""
    for rel, exp in relationships:
        if rel == 'power':
            field_name = f'soil__{rel}_exp{exp}'
            display_name = f'{rel} (exp={exp})'
        else:
            field_name = f'soil__{rel}'
            display_name = rel
            
        create_soil_depth_plot(grid, elevation_field, field_name, display_name)