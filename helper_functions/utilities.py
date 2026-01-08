"""
Important utility functions to run ShallowLandslider separately
"""

# %% Import required packages
from pathlib import Path
import numpy as np
from scipy import stats
import richdem as rd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import pickle

from landlab import RasterModelGrid, imshowhs_grid
from landlab.io import esri_ascii, read_esri_ascii

from bmi_topography import Topography


# %% Getting topography from OpenTopography

def get_topo(
    api_key: str,
    buffer: float,
    north: float = 28.25,
    south: float = 28.23,
    east: float = 85.18,
    west: float = 85.15,
    dem_type: str = "NASADEM",
    smooth_num: int = 0,
    load_dem: str = None,
    verbose: bool = False,
):
    """
    Downloads DEM from OpenTopo and generates a Landlab RasterModelGrid.

    Parameters
    ----------
    api_key : str
        API key for OpenTopo.
    buffer : float
        Additional space around the DEM to remove potential edge effects (in degrees).
    north, south, east, west : float
        Bounding box in decimal degrees.
    dem_type : str
        DEM type from OpenTopo (e.g., NASADEM, SRTMGL1).
    smooth_num : int
        Number of smoothing iterations (0 for none).
    load_dem : str, optional
        Path to local DEM file in ESRI ASCII format.
    verbose : bool
        Print debug information.

    Returns
    -------
    grid : RasterModelGrid
        Landlab grid with elevation field.
    z_geog : ndarray
        Elevation values.
    """
    
    try:
        # If loading local DEM
        if load_dem:
            if not Path(load_dem).exists():
                raise FileNotFoundError(f"DEM file not found: {load_dem}")
            grid, z_geog = read_esri_ascii(load_dem, name="topographic__elevation", halo=1)
            grid.set_nodata_nodes_to_closed(grid.at_node["topographic__elevation"], -9999)
            grid.at_node["topographic__elevation"][
                grid.at_node["topographic__elevation"] == -9999
            ] = np.nan

        else:
            # Validate bounding box
            if north <= south or east <= west:
                raise ValueError("Invalid bounding box: north must be > south and east > west.")
            
            # Prepare OpenTopo request
            params = Topography.DEFAULT.copy()
            params.update({
                "south": south - buffer,
                "north": north + buffer,
                "west": west - buffer,
                "east": east + buffer,
                "dem_type": dem_type,
                "output_format": "AAIGrid",
                "cache_dir": Path.cwd(),
                "api_key": api_key,
            })

            # Fetch DEM
            dem = Topography(**params)
            try:
                name = dem.fetch()
                props = dem.load()
            except Exception as e:
                raise RuntimeError(f"Failed to fetch DEM from OpenTopo: {e}")

            # Load DEM into Landlab
            with open(name) as fp:
                grid_geog = esri_ascii.load(fp, name="topographic__elevation", at="node")
            z_geog = grid_geog.at_node["topographic__elevation"]

            # Extract resolution dynamically
            grid_spacing = props.get("resolution", 30)  # fallback to 30 m if missing

            # Create RasterModelGrid
            grid = RasterModelGrid(
                (grid_geog.number_of_node_rows, grid_geog.number_of_node_columns),
                xy_spacing=grid_spacing,
                xy_axis_units="m",
            )
            grid.add_field("topographic__elevation", z_geog, at="node")

            if verbose:
                print("Request Parameters:", params)
                print("DEM Properties:", props)

        # Apply smoothing if requested
        if smooth_num > 0:
            smoothed_elev = smooth_elevation_grid(grid, method="gaussian", smooth_num=smooth_num)
            grid.at_node["topographic__elevation"] = smoothed_elev
            z_geog = smoothed_elev

        return grid, z_geog

    except Exception as e:
        raise RuntimeError(f"Error in get_topo: {e}")

    # if load_dem is not None:
    #     grid, z_geog = read_esri_ascii(load_dem, name="topographic__elevation",
    #                         halo=1,)
    #     grid.set_nodata_nodes_to_closed(grid.at_node["topographic__elevation"], -9999)
    #     grid.at_node["topographic__elevation"][grid.at_node["topographic__elevation"]==-9999] = np.nan
    # else:
    #     params = Topography.DEFAULT.copy()
    #     params["south"] = south - buffer
    #     params["north"] = north + buffer
    #     params["west"] = west - buffer
    #     params["east"] = east + buffer
    #     params["dem_type"] = dem_type
    #     params["output_format"] = "AAIGrid"
    #     params["cache_dir"] = Path.cwd()
    #     params["api_key"] = api_key
    #     dem = Topography(**params)
    #     name = dem.fetch()
    #     props = dem.load()

    #     with open(name) as fp:
    #         grid_geog = esri_ascii.load(fp, name="topographic__elevation", at="node")

    #     z_geog = grid_geog.at_node["topographic__elevation"]

    #     match dem_type:
    #         case "SRTMGL3" | "COP90":
    #             grid_spacing = 90
    #         case "SRTMGL1" | "AW3D30" | "NASADEM" | "COP30":
    #             grid_spacing = 30

    #     grid = RasterModelGrid(
    #         (grid_geog.number_of_node_rows, grid_geog.number_of_node_columns),
    #         xy_spacing=grid_spacing,
    #         xy_axis_units="m",
    #     )
    #     grid.add_field("topographic__elevation", z_geog, at="node")
        
    #     if verbose:
    #         print(params)
    #         print(props)
    
    # if smooth_num > 0:
    #     # Smooth the downloaded DEM
    #     smoothed_elev = smooth_elevation_grid(
    #         grid,
    #         method="gaussian",
    #         smooth_num=smooth_num,
    #     )
    #     grid.at_node["topographic__elevation"] = smoothed_elev
    #     z_geog = smoothed_elev

    # return grid, z_geog

# %% Smoothen landlab grid
def smooth_elevation_grid(grid, method="mean", smooth_num=1):
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
    if method == "mean":
        from scipy.ndimage import uniform_filter
    elif method == "gaussian":
        # For Gaussian smoothing
        from scipy.ndimage import gaussian_filter
    else:
        raise ValueError("Method must be 'mean' or 'gaussian'")

    # Get the original elevation data
    elevation = grid.at_node["topographic__elevation"].copy()

    # Reshape to 2D for smoothing (assuming a raster grid)
    elevation_2d = elevation.reshape(grid.shape)

    smooth_round = 0
    while smooth_round < smooth_num:
        if method == "mean":
            # Apply a 3x3 mean filter
            smoothed_elevation = uniform_filter(elevation_2d, size=3, mode="reflect")
        elif method == "gaussian":
            # For Gaussian smoothing
            smoothed_elevation = gaussian_filter(elevation_2d, sigma=1, mode="reflect")
        smooth_round += 1

    # Reshape back to 1D for use with landlab
    smoothed_elevation_1d = smoothed_elevation.flatten()

    return smoothed_elevation_1d


# %% Apply soil depth to DEM


def apply_soil_depth(
    grid,
    elevation_field="topographic__elevation",
    soil_field="soil__depth",
    max_soil_depth=1.5,
    distribution="uniform",
    relationship="linear",
    plot=False,
    verbose: bool = False,
    **kwargs,
):
    """
    Apply soil depth to core nodes based on elevation, uniform distribution, or curvature.

    Parameters:
    -----------
    grid : Landlab grid object
        The landlab grid
    elevation_field : str
        Name of the field containing elevation data
    soil_field : str
        Name of the field to store soil depth data
    max_soil_depth : float
        Maximum soil depth in meters (used differently by each method)
    distribution : {'uniform', 'elevation', 'curvature', 'drainage_area'}
        Soil depth distribution method
    relationship : str
        Relationship for 'elevation' or 'curvature' distributions
    plot : bool
        If True, creates a scatter plot showing soil depth vs controlling variable
    **kwargs : dict
        Additional keyword arguments controlling the behavior of each distribution:

        Elevation-based (distribution='elevation')
        ------------------------------------------
        decay_rate : float, default=5.0
            Decay rate for 'exponential' relationship.
        exponent : float, default=1.0
            Exponent for 'power' relationship.

        Drainage-area based (distribution='drainage_area')
        --------------------------------------------------
        drainage_transform : {'log', 'sqrt', 'power', 'threshold', 'linear'}, default='log'
            Transformation of drainage area to normalize magnitudes.
        drainage_power : float, default=0.3
            Exponent for 'power' transform.
        drainage_threshold : float, optional
            Threshold drainage area for 'threshold' transform.
            If None, uses 75th percentile or 100 cells as default.

        Curvature-based (distribution='curvature')
        ------------------------------------------
        relationship : {'linear', 'linear_std_global', 'linear_std_local', 'piecewise'}
            Relationship between curvature (κ) and soil depth.
        a : float, optional
            Intercept term for linear forms (default: max_soil_depth/2).
        window : int, default=5
            Neighborhood size (in pixels) for local curvature std (used in 'linear_std_local').

        Piecewise curvature (relationship='piecewise')
        ----------------------------------------------
        P0 : float, default=1.0
            Reference curvature scaling.
        h_star : float, default=1.0
            Scaling constant for log law.
        D : float, default=1.0
            Diffusivity-like term.
        h_min : float, default=0.1
            Minimum soil depth when κ ≤ 0.
        h_no_ss : float, default=0.0
            Soil depth for nodes with no steady-state (e.g., bare rock).
        eps : float, default=1e-10
            Small number to avoid log(0).

    Returns
    -------
    soil_depth : ndarray
        1D array of soil depth values, associated with the grid 'soil__depth' field.

    Notes
    -----
    - For curvature relationships based on Patton et al. (2018):
        slope b = -446.3 * std(κ) + 30.3
    where std(κ) is either global (all core nodes) or local (moving window).
    """
    elev_depth = None
    curv_depth = None

    elevation = grid.at_node[elevation_field]
    core_nodes = grid.core_nodes

    # Initialize soil depth field (zeros)
    soil_depth = grid.add_zeros("node", soil_field, clobber=True)

    if distribution == "uniform":
        soil_depth[core_nodes] = max_soil_depth
        if verbose:
            print(
                f"Uniform soil depth applied: {max_soil_depth:.2f} m to {len(core_nodes)} core nodes."
            )

    elif distribution == "elevation":
        core_elevations = elevation[core_nodes]
        min_elevation = np.min(core_elevations)
        max_elevation_core = np.max(core_elevations)
        elevation_range = max_elevation_core - min_elevation

        if elevation_range == 0:
            soil_depth[core_nodes] = max_soil_depth / 2.0
            if verbose:
                print(
                    "Warning: All core nodes have the same elevation. Assigning uniform soil depth."
                )
        else:
            normalized_elevation = (elevation - min_elevation) / elevation_range
            if relationship == "linear":
                temp_soil_depth = max_soil_depth * (1.0 - normalized_elevation)
            elif relationship == "exponential":
                decay_rate = kwargs.get("decay_rate", 5.0)
                top_term = np.exp(-decay_rate)
                temp_soil_depth = (
                    max_soil_depth
                    * (np.exp(-decay_rate * normalized_elevation) - top_term)
                    / (1 - top_term)
                )
            elif relationship == "power":
                exponent = kwargs.get("exponent", 1.0)
                temp_soil_depth = (
                    max_soil_depth * (1.0 - normalized_elevation) ** exponent
                )
            elif relationship == "sigmoid":
                sigmoid_input = 6.0 * (normalized_elevation - 0.5)
                sigmoid_output = 1.0 / (1.0 + np.exp(sigmoid_input))
                temp_soil_depth = max_soil_depth * sigmoid_output
            else:
                raise ValueError(f"Unknown relationship type: {relationship}")

            temp_soil_depth = np.maximum(temp_soil_depth, 0.0)
            soil_depth[:] = temp_soil_depth
            soil_depth[np.setdiff1d(np.arange(grid.number_of_nodes), core_nodes)] = 0.0

        if verbose:
            print(
                f"Elevation-based soil depth applied: elevation range {min_elevation:.2f}–{max_elevation_core:.2f} m."
            )

    elif distribution == "curvature":
        # Ensure curvature exists
        if "curvature" not in grid.at_node:
            kappa_map = calculate_terrain_attribute(
                grid=grid,
                field_name=elevation_field,
                attrib="planform_curvature",
                out_field="curvature",
            )
        else:
            kappa_map = grid.at_node["curvature"]

        core_kappa = kappa_map[core_nodes]
        min_kappa, max_kappa = np.nanmin(core_kappa), np.nanmax(core_kappa)

        if max_kappa == min_kappa:
            raise ValueError("No variation in curvature. Check inputs")

        a = kwargs.get("a", max_soil_depth / 2.0)
        scale = kwargs.get("scale", 0.01)  # default scaling for Patton regression

        if relationship == "linear":
            # Plain linear
            b = kwargs.get("b", -1.0)
            soil_depth[core_nodes] = np.clip(
                a + b * core_kappa, 0.0, kwargs.get("max_soil_depth", 1.5)
            )

        elif relationship == "piecewise":
            # Piecewise log-law function
            P0 = kwargs.get("P0", 1.0)
            h_star = kwargs.get("h_star", 1.0)
            D = kwargs.get("D", 1.0)
            h_min = kwargs.get("h_min", 0.1)
            h_no_ss = kwargs.get("h_no_ss", 0.0)
            eps = kwargs.get("eps", 1e-10)

            h_vals = np.zeros_like(core_kappa, dtype=float)
            for i, kappa in enumerate(core_kappa):
                if kappa <= 0:
                    h_vals[i] = h_min
                else:
                    s = -D * kappa / P0
                    if 0 < s < 1:
                        h_vals[i] = -h_star * np.log(s + eps)
                    elif s >= 1:
                        h_vals[i] = h_no_ss
                    else:
                        h_vals[i] = h_min
            soil_depth[core_nodes] = np.clip(
                h_vals, 0.0, kwargs.get("max_soil_depth", 1.5)
            )

        elif relationship == "linear_std_global":
            # Patton-style global std regression
            curv_std = np.nanstd(core_kappa)
            b = (-446.3 * curv_std + 30.3) * scale
            soil_depth[core_nodes] = np.clip(
                a + b * core_kappa, 0.0, kwargs.get("max_soil_depth", 1.5)
            )
            if verbose:
                print(f"Global curvature std: {curv_std:.3e}, slope b={b:.3f}")

        elif relationship == "linear_std_local":
            # Patton-style local std regression
            from scipy.ndimage import generic_filter

            window = kwargs.get("window", 5)
            nrows, ncols = grid.shape
            kappa2d = kappa_map.reshape((nrows, ncols))

            # Compute local std deviation of curvature
            kappa_std2d = generic_filter(kappa2d, np.std, size=window, mode="nearest")

            # Compute local slope using Patton regression
            local_b = (-446.3 * kappa_std2d + 30.3) * scale

            # Extract only core nodes
            core_local_b = local_b.ravel()[core_nodes]

            soil_depth[core_nodes] = np.clip(
                a + core_local_b * core_kappa, 0.0, kwargs.get("max_soil_depth", 1.5)
            )

            if verbose:
                print(
                    f"Local curvature std window={window}, mean slope={np.nanmean(core_local_b):.3f}"
                )

        else:
            raise ValueError(f"Unknown curvature relationship: {relationship}")
        if verbose:
            print(
                f"Curvature-based soil depth applied with '{relationship}' relationship."
            )
            print(
                f"Soil depth range: {np.min(soil_depth[core_nodes]):.3f} – {np.max(soil_depth[core_nodes]):.3f} m"
            )

    elif distribution == "drainage_area":
        if "drainage_area" not in grid.at_node:
            raise ValueError(
                "Drainage area field not found in grid. Run flow routing first."
            )

        core_drainage_area = grid.at_node["drainage_area"][core_nodes]
        grid_area = grid.dx**2

        # Handle zero or very small drainage areas
        min_drainage = np.maximum(
            np.min(core_drainage_area[core_drainage_area > 0]), 1e-10
        )
        core_drainage_area = np.maximum(core_drainage_area, min_drainage)

        # Apply transformation to handle multiple orders of magnitude
        drainage_transform = kwargs.get("drainage_transform", "log")
        if drainage_transform == "log":
            # Logarithmic transformation
            log_drainage = np.log10(core_drainage_area)
            min_log = np.min(log_drainage)
            max_log = np.max(log_drainage)
            if max_log > min_log:
                normalized_drainage = (log_drainage - min_log) / (max_log - min_log)
            else:
                normalized_drainage = np.ones_like(log_drainage) * 0.5
            if verbose:
                print(f"Log drainage area range: {min_log:.2f} to {max_log:.2f}")

        elif drainage_transform == "sqrt":
            # Square root transformation
            sqrt_drainage = np.sqrt(core_drainage_area)
            min_sqrt = np.min(sqrt_drainage)
            max_sqrt = np.max(sqrt_drainage)
            if max_sqrt > min_sqrt:
                normalized_drainage = (sqrt_drainage - min_sqrt) / (max_sqrt - min_sqrt)
            else:
                normalized_drainage = np.ones_like(sqrt_drainage) * 0.5
            if verbose:
                print(f"Sqrt drainage area range: {min_sqrt:.2f} to {max_sqrt:.2f}")

        elif drainage_transform == "power":
            drainage_power = kwargs.get("drainage_power", 0.3)
            # Power transformation (similar to sqrt but adjustable)
            power_drainage = core_drainage_area**drainage_power
            min_power = np.min(power_drainage)
            max_power = np.max(power_drainage)
            if max_power > min_power:
                normalized_drainage = (power_drainage - min_power) / (
                    max_power - min_power
                )
            else:
                normalized_drainage = np.ones_like(power_drainage) * 0.5
            if verbose:
                print(
                    f"Power ({drainage_power}) drainage area range: {min_power:.2f} to {max_power:.2f}"
                )

        elif drainage_transform == "threshold":
            drainage_threshold = kwargs.get("drainage_threshold", None)
            # Threshold-based approach
            if drainage_threshold is None:
                # Use median as default threshold
                drainage_threshold = max(
                    grid_area * 100, np.percentile(core_drainage_area, 75)
                )

            # Binary classification: above threshold gets max depth, below gets scaled
            above_threshold = core_drainage_area >= drainage_threshold
            normalized_drainage = np.zeros_like(core_drainage_area)

            # Nodes above threshold → maximum soil depth
            normalized_drainage[above_threshold] = 1.0

            # Nodes below threshold → log-scaled depth
            below_mask = ~above_threshold
            if np.any(below_mask):
                below_drainage = core_drainage_area[below_mask]
                log_below = np.log10(below_drainage)
                min_log = np.min(log_below)
                max_log = np.log10(drainage_threshold)

                # Normalize log values to [0, 1), leaving 1.0 for above-threshold nodes
                norm_vals = (log_below - min_log) / (max_log - min_log)
                normalized_drainage[below_mask] = np.clip(norm_vals, 0.0, 0.999)

            if verbose:
                print(f"Threshold drainage area: {drainage_threshold:.2f}")
                print(f"Nodes above threshold: {np.sum(above_threshold)}")

        elif drainage_transform == "linear":
            # Original linear approach (for comparison)
            max_drainage_area = np.max(core_drainage_area)
            if max_drainage_area > 0:
                normalized_drainage = core_drainage_area / max_drainage_area
            else:
                normalized_drainage = np.ones_like(core_drainage_area) * 0.5
            if verbose:
                print(f"Linear drainage area range: 0 to {max_drainage_area:.2f}")

        else:
            raise ValueError(f"Unknown drainage_transform: {drainage_transform}")

        # Apply soil depth
        soil_depth[core_nodes] = normalized_drainage * max_soil_depth

        if verbose:
            print(
                f"Drainage area-based soil depth applied using '{drainage_transform}' transformation."
            )
            print(
                f"Soil depth range: {np.min(soil_depth[core_nodes]):.3f} to {np.max(soil_depth[core_nodes]):.3f} m"
            )

    elif distribution == "mean_elev_curv":
        if verbose:
            print(
                "Computing composite soil depth: mean of elevation and curvature-based soil depth"
            )
        elev_depth = apply_soil_depth(
            grid,
            elevation_field="topographic__elevation",
            soil_field=soil_field,
            max_soil_depth=max_soil_depth,
            distribution="elevation",
            relationship="linear",
            plot=False,
        )
        curv_depth = apply_soil_depth(
            grid,
            elevation_field="topographic__elevation",
            soil_field=soil_field,
            max_soil_depth=max_soil_depth,
            distribution="curvature",
            relationship="linear_std_local",
            plot=False,
        )

        soil_depth[:] = np.mean([elev_depth, curv_depth], axis=0)
    else:
        raise ValueError(
            "Soil distribution must be 'uniform', 'elevation', 'curvature, 'drainage_area', or 'mean_elev_curv'."
        )

    if plot:
        create_soil_depth_plot(
            grid,
            elevation_field,
            soil_field,
            distribution,
            relationship,
            elev_component=elev_depth,
            curv_component=curv_depth,
        )

    return soil_depth


# %%% Helper functions for applying soil depth
def calculate_terrain_attribute(grid, field_name, attrib, out_field=None):
    """
    Compute a terrain attribute with richdem and add it to a Landlab grid.
    Automatically detects nodata values from the field.

    Parameters
    ----------
    mg : landlab.ModelGrid
        The Landlab grid (must be RasterModelGrid for reshaping).
    field_name : str
        Name of the Landlab field containing elevation (e.g. 'topographic__elevation').
    attrib : str
        Richdem terrain attribute to calculate (e.g. 'slope_riserun', 'curvature').
    out_field : str, optional
        Name of the output field in Landlab (defaults to attrib name).

    Returns
    -------
    numpy.ndarray
        The computed attribute values (1D, node order).
    """
    if out_field is None:
        out_field = attrib

    # Grab field from landlab
    z = grid.at_node[field_name]
    nrows, ncols = grid.shape
    dem2d = z.reshape((nrows, ncols))

    # Detect nodata: if any NaNs, set nodata to np.nan, else use -9999
    if np.isnan(dem2d).any():
        nodata = np.nan
    else:
        nodata = -9999  # fallback

    # Wrap into richdem rdarray
    dem_rd = rd.rdarray(dem2d.copy(), no_data=nodata)
    dem_rd.geotransform = [0, grid.dx, 0, 0, 0, -grid.dy]

    # Compute terrain attribute with richdem
    result2d = rd.TerrainAttribute(dem_rd, attrib=attrib)

    # Flatten to Landlab node order
    result1d = np.asarray(result2d).ravel()

    # Add to Landlab grid
    grid.add_field(out_field, result1d, at="node", clobber=True)

    return result1d


def create_soil_depth_plot(
    grid,
    elevation_field,
    soil_field,
    distribution,
    relationship,
    elev_component=None,
    curv_component=None,
):
    """
    Create scatter plots showing the relationship between soil depth and
    elevation (and drainage area if applicable).
    """

    # Get data for core nodes only
    elevation = grid.at_node[elevation_field]
    soil_depth = grid.at_node[soil_field]
    core_nodes = grid.core_nodes

    core_elevation = elevation[core_nodes]
    core_soil_depth = soil_depth[core_nodes]

    if distribution == "drainage_area":
        if "drainage_area" not in grid.at_node:
            raise ValueError("Drainage area field not found. Run flow routing first.")
        drainage_area = grid.at_node["drainage_area"][core_nodes]

        fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")

        # Subplot 1: Soil depth vs drainage area
        axes[0].scatter(
            drainage_area,
            core_soil_depth,
            alpha=0.6,
            s=30,
            color="blue",
            label=f"Core nodes (n={len(core_nodes)})",
        )
        axes[0].set_xlabel("Drainage Area (m²)")
        axes[0].set_xscale("log")
        axes[0].set_ylabel("Soil Depth (m)")
        axes[0].set_title("Soil Depth vs Drainage Area")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Subplot 2: Soil depth vs elevation
        axes[1].scatter(
            core_elevation,
            core_soil_depth,
            alpha=0.6,
            s=30,
            color="green",
            label=f"Core nodes (n={len(core_nodes)})",
        )
        axes[1].set_xlabel("Elevation (m)")
        axes[1].set_ylabel("Soil Depth (m)")
        axes[1].set_title("Soil Depth vs Elevation")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Stats box on second subplot
        stats_text = (
            f"Elevation: {core_elevation.min():.2f} - {core_elevation.max():.2f} m\n"
            f"Soil depth: {core_soil_depth.min():.2f} - {core_soil_depth.max():.2f} m\n"
            f"Mean soil depth: {core_soil_depth.mean():.2f} m\n"
            f"Drainage area: {drainage_area.min():.2f} - {drainage_area.max():.2f} m²"
        )
        axes[1].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[1].transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Subplot 3: Histogram of drainage area with percentile markers
        log_drainage = np.log10(drainage_area)

        axes[2].hist(log_drainage, bins=50, color="purple", alpha=0.7)
        axes[2].set_xlabel("log10 Drainage Area (m²)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Distribution of log10(Drainage Area)")
        axes[2].grid(True, alpha=0.3)

        # Percentiles to mark
        percentiles = [25, 50, 75, 90, 95]
        perc_values = np.percentile(log_drainage, percentiles)

        for p, val in zip(percentiles, perc_values):
            axes[2].axvline(val, color="red", linestyle="--", alpha=0.7)
            axes[2].text(
                val,
                axes[2].get_ylim()[1] * 0.9,
                f"{p}%",
                rotation=90,
                va="top",
                ha="center",
                color="red",
                fontsize=8,
            )

        plt.show()

    elif distribution == "curvature":
        core_kappa = grid.at_node["curvature"][core_nodes]

        fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained")

        # Panel 1: Soil depth vs curvature
        axes[0].scatter(core_kappa, core_soil_depth, alpha=0.6, s=30, color="blue")
        axes[0].set_xlabel("Curvature (κ)")
        axes[0].set_ylabel("Soil Depth (m)")
        axes[0].set_title("Soil Depth vs Curvature")
        axes[0].grid(True, alpha=0.3)

        # Panel 2: Soil depth vs elevation
        axes[1].scatter(core_elevation, core_soil_depth, alpha=0.6, s=30, color="green")
        axes[1].set_xlabel("Elevation (m)")
        axes[1].set_ylabel("Soil Depth (m)")
        axes[1].set_title("Soil Depth vs Elevation")
        axes[1].grid(True, alpha=0.3)

        # Panel 3: Histogram of curvature
        axes[2].hist(core_kappa, bins=50, color="purple", alpha=0.7)
        axes[2].set_xlabel("Curvature (κ)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Distribution of Curvature")
        axes[2].grid(True, alpha=0.3)

        # Optional: mark percentiles
        percentiles = [25, 50, 75, 90, 95]
        perc_values = np.percentile(core_kappa, percentiles)
        for p, val in zip(percentiles, perc_values):
            axes[2].axvline(val, color="red", linestyle="--", alpha=0.7)
            axes[2].text(
                val,
                axes[2].get_ylim()[1] * 0.9,
                f"{p}%",
                rotation=90,
                va="top",
                ha="center",
                color="red",
                fontsize=8,
            )

    elif distribution == "mean_elev_curv":
        if elev_component is None or curv_component is None:
            raise ValueError(
                "Composite plot requires 'elev_component' and 'curv_component' arrays."
            )

        if "curvature" not in grid.at_node:
            raise ValueError("Curvature field not found on grid.")

        core_kappa = grid.at_node["curvature"][core_nodes]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")

        # --- Panel 1: Elevation-based component ---
        axes[0].scatter(
            core_elevation, elev_component[core_nodes], alpha=0.6, s=30, color="green"
        )
        axes[0].set_xlabel("Elevation (m)")
        axes[0].set_ylabel("Soil Depth (m)")
        axes[0].set_title("Elevation-based Soil Depth")
        axes[0].grid(True, alpha=0.3)

        # --- Panel 2: Curvature-based component ---
        axes[1].scatter(
            core_kappa, curv_component[core_nodes], alpha=0.6, s=30, color="blue"
        )
        axes[1].set_xlabel("Curvature (κ)")
        axes[1].set_ylabel("Soil Depth (m)")
        axes[1].set_title("Curvature-based Soil Depth")
        axes[1].grid(True, alpha=0.3)

        # --- Panel 3: Final combined (mean) result ---
        axes[2].scatter(
            core_elevation, core_soil_depth, alpha=0.6, s=30, color="purple"
        )
        axes[2].set_xlabel("Elevation (m)")
        axes[2].set_ylabel("Soil Depth (m)")
        axes[2].set_title("Final Composite (Mean) Soil Depth")
        axes[2].grid(True, alpha=0.3)

        # Summary stats box
        stats_text = (
            f"Elev-comp mean: {np.nanmean(elev_component[core_nodes]):.2f} m\n"
            f"Curv-comp mean: {np.nanmean(curv_component[core_nodes]):.2f} m\n"
            f"Final mean: {np.nanmean(core_soil_depth):.2f} m\n"
            f"Range: {np.nanmin(core_soil_depth):.2f} – {np.nanmax(core_soil_depth):.2f} m"
        )
        axes[0].text(
            0.02,
            0.98,
            stats_text,
            transform=axes[0].transAxes,
            verticalalignment="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        fig.suptitle(
            "Diagnostic: Elevation vs Curvature vs Composite Soil Depth", fontsize=12
        )
        plt.show()

    else:
        # Original single-plot behavior
        plt.figure(layout="constrained")
        if distribution == "uniform":
            plt.scatter(
                core_elevation,
                core_soil_depth,
                alpha=0.6,
                s=30,
                color="blue",
                label=f"Core nodes (n={len(core_nodes)})",
            )
            plt.axhline(
                y=core_soil_depth[0],
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Uniform depth = {core_soil_depth[0]:.2f} m",
            )
            plt.title("Uniform Soil Depth Distribution")
        else:
            plt.scatter(
                core_elevation,
                core_soil_depth,
                alpha=0.6,
                s=30,
                color="blue",
                label=f"Core nodes (n={len(core_nodes)})",
            )
            plt.title(f"{relationship.capitalize()}-based soil depth distribution")

            elev_range = np.linspace(core_elevation.min(), core_elevation.max(), 100)
            max_depth = core_soil_depth.max()
            min_elev = core_elevation.min()
            max_elev = core_elevation.max()

            if max_elev > min_elev:  # Avoid division by zero
                normalized_elev = (elev_range - min_elev) / (max_elev - min_elev)
                theoretical_depth = max_depth * (1.0 - normalized_elev)
                plt.plot(
                    elev_range,
                    theoretical_depth,
                    "r--",
                    alpha=0.7,
                    label="Theoretical relationship",
                )

        plt.xlabel("Elevation (m)")
        plt.ylabel("Soil Depth (m)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        stats_text = (
            f"Elevation: {core_elevation.min():.2f} - {core_elevation.max():.2f} m\n"
            f"Soil depth: {core_soil_depth.min():.2f} - {core_soil_depth.max():.2f} m\n"
            f"Mean soil depth: {core_soil_depth.mean():.2f} m"
        )
        plt.text(
            0.02,
            0.22,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()


# %% Generate earthquakes
def generate_acceleration_grid(
    grid,
    horizontal_max,
    vertical_max,
    distribution="uniform",
    center=None,
    random_center=False,
    seed=None,
    plot_grids=False,
):
    """
    Generate arrays of horizontal and vertical acceleration values for a landlab grid.

    Parameters:
    ----------
    grid : RasterModelGrid
        The landlab grid to generate acceleration values for
    max_horizontal : float
        Maximum horizontal acceleration value at the center
    max_vertical : float
        Maximum vertical acceleration value at the center
    distribution : str, optional
        Distribution shape: "uniform", "circular", "square", "diamond", or "exponential" (default: "uniform")
    center : tuple, optional
        (row, col) coordinates of the center point. If None and random_center is False,
        the center of the grid is used.
    random_center : bool, optional
        If True, a random center point will be selected. This overrides the center parameter.
    seed : int, optional
        Random seed for reproducibility when using random_center

    Returns:
    -------
    tuple
        (horizontal_acceleration, vertical_acceleration) - Two numpy arrays with values at grid nodes
    """
    if distribution != "uniform":
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Get grid dimensions
        rows, cols = grid.shape

        # Initialize arrays for node values
        num_nodes = grid.number_of_nodes
        horizontal_accel = np.zeros(num_nodes)
        vertical_accel = np.zeros(num_nodes)

        # Get x and y coordinates of all nodes
        node_x = grid.x_of_node
        node_y = grid.y_of_node

        # Create mask of valid (non-NaN) nodes
        valid_mask = ~np.isnan(node_x) & ~np.isnan(node_y)

        # Calculate center coordinates
        if random_center:
            # Get indices of valid nodes
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                # Fallback if no valid nodes (shouldn't happen)
                valid_indices = np.arange(num_nodes)

            # Select a random valid node as center
            center_node = np.random.choice(valid_indices)
            center_x = node_x[center_node]
            center_y = node_y[center_node]

            # Try to get row, col for reporting
            try:
                center_row, center_col = grid.node_row_col(center_node)

                print(
                    f"Randomly selected center at row={center_row}, col={center_col} (node={center_node})"
                )
            except:
                print(
                    f"Randomly selected center at node={center_node}, x={center_x}, y={center_y}"
                )

        elif center is None:
            # Use geometric center of valid nodes
            valid_x = node_x[valid_mask]
            valid_y = node_y[valid_mask]

            if len(valid_x) > 0:
                center_x = np.mean(valid_x)
                center_y = np.mean(valid_y)
            else:
                # Fallback to all nodes (unlikely scenario)
                center_x = np.mean(node_x[~np.isnan(node_x)])
                center_y = np.mean(node_y[~np.isnan(node_y)])
        else:
            # Center is provided as (row, col)
            center_row, center_col = center
            # Convert row, col to node ID
            center_row = min(max(0, center_row), rows - 1)
            center_col = min(max(0, center_col), cols - 1)
            center_node = grid.grid_coords_to_node_id(center_row, center_col)
            center_x = node_x[center_node]
            center_y = node_y[center_node]

        # Calculate normalization factors based on distribution
        # Use only valid nodes for distance calculations
        valid_x = node_x[valid_mask]
        valid_y = node_y[valid_mask]

        if distribution == "circular" or distribution == "exponential":
            # Euclidean distance
            distances = np.sqrt((valid_x - center_x) ** 2 + (valid_y - center_y) ** 2)
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
        elif distribution == "square":
            # Chebyshev distance (max of x/y distances)
            x_distances = np.abs(valid_x - center_x)
            y_distances = np.abs(valid_y - center_y)
            max_distance = max(
                np.max(x_distances) if len(x_distances) > 0 else 1.0,
                np.max(y_distances) if len(y_distances) > 0 else 1.0,
            )
        elif distribution == "diamond":
            # Manhattan distance (sum of x/y distances)
            x_distances = np.abs(valid_x - center_x)
            y_distances = np.abs(valid_y - center_y)
            distances = x_distances + y_distances
            max_distance = np.max(distances) if len(distances) > 0 else 1.0

        # Ensure max_distance is not zero
        max_distance = max(max_distance, 1e-10)

        # Calculate acceleration values for each node
        for i in range(num_nodes):
            # Skip nodes with NaN coordinates
            if np.isnan(node_x[i]) or np.isnan(node_y[i]):
                horizontal_accel[i] = 0.0
                vertical_accel[i] = 0.0
                continue

            # Calculate distance from center
            dx = node_x[i] - center_x
            dy = node_y[i] - center_y

            if distribution == "circular":
                distance = np.sqrt(dx**2 + dy**2)
                factor = max(0.0, 1.0 - distance / max_distance)
            elif distribution == "square":
                distance = max(abs(dx), abs(dy))
                factor = max(0.0, 1.0 - distance / max_distance)
            elif distribution == "diamond":
                distance = abs(dx) + abs(dy)  # Manhattan distance
                factor = max(0.0, 1.0 - distance / max_distance)
            elif distribution == "exponential":
                distance = np.sqrt(dx**2 + dy**2)
                # Scale factor for exponential decay
                decay_factor = 3.0  # Adjust for faster/slower decay
                factor = np.exp(-distance / (max_distance / decay_factor))

            # Set the acceleration values
            horizontal_accel[i] = horizontal_max * factor
            vertical_accel[i] = vertical_max * factor
    else:
        horizontal_accel = np.ones_like(grid.at_node["topographic__elevation"])
        vertical_accel = np.ones_like(grid.at_node["topographic__elevation"])

        horizontal_accel[np.isnan(grid.at_node["topographic__elevation"])] = np.nan
        vertical_accel[np.isnan(grid.at_node["topographic__elevation"])] = np.nan

        horizontal_accel[grid.core_nodes] *= horizontal_max
        vertical_accel[grid.core_nodes] *= vertical_max

    if plot_grids:
        # Plot PGA arrays
        plt.figure(layout="constrained")
        plt.subplot(1, 2, 1)
        imshowhs_grid(
            grid,
            "topographic__elevation",
            plot_type="Drape1",
            drape1=horizontal_accel,
            cmap="Reds",
            allow_colorbar=True,
            ticks_km=True,
        )

        plt.subplot(1, 2, 2)
        imshowhs_grid(
            grid,
            "topographic__elevation",
            plot_type="Drape1",
            drape1=vertical_accel,
            cmap="Reds",
            allow_colorbar=True,
            ticks_km=True,
        )

        plt.suptitle("Earthquake PGA (in multiples of g)")
        plt.show()

    print(f"{distribution} horizontal and vertical PGA arrays generated")
    return horizontal_accel, vertical_accel


# %% Load measured data
"""
Loads measured data from stored pickles or creates new pickles from formatted csv files and loads those

Measured data is used for statistically-based region splitting and data comparison 
"""


def pickle_or_not_to_pickle(
    file_name_dict, pickle_path="measured_data.pkl", min_area=900,
    verbose: bool = False
):
    """
    Load processed data (DataFrames, shapefile, KDEs) from pickle if it exists.
    Otherwise, build from source files, save, and return.
    """

    if os.path.exists(pickle_path):
        if verbose:
            print(f"Loading preprocessed data from {pickle_path}...")
        with open(pickle_path, "rb") as f:
            bundle = pickle.load(f)
        return bundle
    if verbose:
        print("Pickle not found, building from CSVs and shapefile...")

    # --- Load CSVs ---
    # All measured landslide areas
    measured_data = pd.read_csv(file_name_dict["file1"])

    # All measured landslide zonal statistics (elevation, slope, aspect)
    measured_spatial_stats = pd.read_csv(file_name_dict["file2"])

    # Filter out landslides below sensitivity threshold
    measured_spatial_stats_900greater = measured_spatial_stats.drop(
        measured_spatial_stats[measured_spatial_stats["Area"] <= min_area].index
    )

    # Measured landslide zonal statistics inside selected area
    measured_spatial_stats_clipped = pd.read_csv(file_name_dict["file3"])

    # --- Fit KDE ---
    kde_data, kde_transform = fit_bivariate_kde(
        dataframe=measured_data,
        x_col="length_m",
        y_col="width_m",
        category_col=None,
        plot_results=False,
    )

    # --- Bundle everything ---
    bundle = {
        "measured_data": measured_data,
        "measured_spatial_stats": measured_spatial_stats,
        "measured_spatial_stats_clipped": measured_spatial_stats_clipped,
        "measured_spatial_stats_900greater": measured_spatial_stats_900greater,
        "kde_data": kde_data,
        "kde_transform": kde_transform,
    }

    # Save to pickle for next time
    with open(pickle_path, "wb") as f:
        pickle.dump(bundle, f)
    if verbose:
        print(f"Saved preprocessed data to {pickle_path}")

    return bundle


# %%% Bivariate kde fitting for region splitting
def fit_bivariate_kde(
    dataframe,
    x_col,
    y_col,
    category_col=None,
    log_transform=True,
    bandwidth=None,
    n_levels=20,
    cmap="viridis",
    figsize=(12, 10),
    plot_results=True,
):
    """
    Create bivariate KDEs for overall data and by category, and return KDE objects for sampling.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing the bivariate data
    x_col, y_col : str
        Names of the columns for the variables
    category_col : str or None
        Name of the categorical column to group by (None for no grouping)
    log_transform : bool or tuple
        If/which variables to log-transform (True for both, or tuple for selective)
    bandwidth : float, array-like, or dict
        Bandwidth for the KDE (can be different per group if dict)
    n_levels : int
        Number of contour levels for the plot
    cmap : str
        Colormap for the contour plot
    figsize : tuple
        Figure size for plots
    plot_results : bool
        Whether to generate plots (set to False to only return KDE objects)

    Returns:
    --------
    kde_results : dict
        Dictionary with 'overall' KDE and group-specific KDEs if categorical variable provided
    transform_info : dict
        Information about the transformation for use in sampling
    """
    # Make a copy to avoid modifying the original dataframe
    x_vals = np.array(dataframe[x_col])
    y_vals = np.array(dataframe[y_col])

    # Set up transformation info
    transform_info = {
        "x_col": x_col,
        "y_col": y_col,
        "log_x": False,
        "log_y": False,
    }

    # Handle log transformation
    if isinstance(log_transform, tuple):
        log_x, log_y = log_transform
    else:
        log_x = log_y = log_transform

    # Apply transformations if requested
    if log_x:
        if x_vals.min() <= 0:
            raise ValueError(f"Cannot log-transform {x_col}: contains values <= 0")
        x_vals = np.log(x_vals)
        transform_info["log_x"] = True

    if log_y:
        if y_vals.min() <= 0:
            raise ValueError(f"Cannot log-transform {y_col}: contains values <= 0")
        y_vals = np.log(y_vals)
        transform_info["log_y"] = True

    # Calculate appropriate bounds for each variable
    # These prevent KDE from extending to unreasonable regions
    if log_x:
        x_min = x_vals.min() - 0.1 * np.abs(x_vals.min())
        x_max = x_vals.max() + 0.1 * np.abs(x_vals.max())
    else:
        x_std = x_vals.std()
        x_min = x_vals.min() - 0.5 * x_std
        x_max = x_vals.max() + 0.5 * x_std

    if log_y:
        y_min = y_vals.min() - 0.1 * np.abs(y_vals.min())
        y_max = y_vals.max() + 0.1 * np.abs(y_vals.max())
    else:
        y_std = y_vals.std()
        y_min = y_vals.min() - 0.5 * y_std
        y_max = y_vals.max() + 0.5 * y_std

    # Store bounds information
    x_bounds = (x_min, x_max)
    y_bounds = (y_min, y_max)
    transform_info["x_bounds"] = x_bounds
    transform_info["y_bounds"] = y_bounds

    # Initialize results dictionary
    kde_results = {}

    # Create overall KDE
    data = np.vstack([x_vals, y_vals])
    kde_overall = stats.gaussian_kde(data, bw_method=bandwidth)
    kde_results["overall"] = kde_overall

    # Group data by category if provided
    if category_col is not None and category_col in dataframe.columns:
        # Get unique categories
        categories = dataframe[category_col].unique()
        transform_info["categories"] = list(categories)

        # Create dictionary to store KDEs by category
        kde_by_category = {}

        # Create KDE for each category
        for category in categories:
            category_mask = dataframe[category_col] == category

            # Skip if too few data points
            if np.sum(category_mask) < 5:
                print(
                    f"Warning: Category '{category}' has fewer than 5 data points. Skipping KDE."
                )
                continue

            cat_x_vals = (
                x_vals[category_mask]
                if isinstance(x_vals, np.ndarray)
                else np.array(x_vals[category_mask])
            )
            cat_y_vals = (
                y_vals[category_mask]
                if isinstance(y_vals, np.ndarray)
                else np.array(y_vals[category_mask])
            )

            cat_data = np.vstack([cat_x_vals, cat_y_vals])

            # Get bandwidth for this category
            cat_bandwidth = bandwidth
            if isinstance(bandwidth, dict):
                cat_bandwidth = bandwidth.get(category, None)

            # Create KDE for this category
            try:
                cat_kde = stats.gaussian_kde(cat_data, bw_method=cat_bandwidth)
                kde_by_category[category] = cat_kde
            except np.linalg.LinAlgError:
                print(
                    f"Warning: Could not create KDE for category '{category}'. Insufficient or collinear data."
                )
                continue

        # Add category KDEs to results
        kde_results["by_category"] = kde_by_category

    # Generate plots if requested
    if plot_results:
        plot_bivariate_kde(
            dataframe,
            x_col,
            y_col,
            category_col,
            kde_results,
            transform_info,
            n_levels=n_levels,
            cmap=cmap,
            figsize=figsize,
        )

    # Return the KDEs and transformation info
    return kde_results, transform_info


# %%%% Plot resulting KDE-matching
def plot_bivariate_kde(
    dataframe,
    x_col,
    y_col,
    category_col=None,
    kde_results=None,
    transform_info=None,
    n_levels=20,
    cmap="viridis",
    figsize=(12, 10),
):
    """
    Plot bivariate KDEs for overall data and by category.

    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing the bivariate data
    x_col, y_col : str
        Names of the columns for the variables
    category_col : str or None
        Name of the categorical column used for grouping
    kde_results : dict
        Dictionary with KDE objects returned by fit_bivariate_kde_with_categories
    transform_info : dict
        Transformation information from fit_bivariate_kde_with_categories
    n_levels : int
        Number of contour levels for the plot
    cmap : str
        Colormap for the contour plot
    figsize : tuple
        Figure size for plots
    """
    if kde_results is None or transform_info is None:
        # Call the main function to get KDE results if not provided
        kde_results, transform_info = fit_bivariate_kde(
            dataframe, x_col, y_col, category_col, plot_results=False
        )

    # Extract transformation info
    log_x = transform_info.get("log_x", False)
    log_y = transform_info.get("log_y", False)
    x_bounds = transform_info.get("x_bounds")
    y_bounds = transform_info.get("y_bounds")

    # Create transformed data for plotting
    x_vals = np.array(dataframe[x_col])
    y_vals = np.array(dataframe[y_col])

    if log_x:
        x_vals = np.log(x_vals)
    if log_y:
        y_vals = np.log(y_vals)

    # Create grid for evaluation
    n_grid = 100
    x_grid = np.linspace(x_bounds[0], x_bounds[1], n_grid)
    y_grid = np.linspace(y_bounds[0], y_bounds[1], n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Plot the overall KDE in transformed space
    plt.figure(figsize=figsize)

    # Plot the transformed data points
    plt.scatter(x_vals, y_vals, alpha=0.3, s=10, color="black")

    # Evaluate and plot overall KDE
    kde_overall = kde_results["overall"]
    Z = kde_overall(positions).reshape(X.shape)

    # Plot the contours
    contour = plt.contourf(X, Y, Z, levels=n_levels, cmap=cmap, alpha=0.8)
    plt.colorbar(contour, label="Density")

    # Add contour lines
    contour_lines = plt.contour(
        X, Y, Z, levels=n_levels, colors="white", linewidths=0.5, alpha=0.5
    )

    # Set axis labels based on transformation
    x_label = f"log({x_col})" if log_x else x_col
    y_label = f"log({y_col})" if log_y else y_col
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title("Overall Bivariate KDE in Transformed Space")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Also show original scale plot with seaborn
    plt.figure(figsize=figsize)

    # Create a joint plot for the original data
    g = sns.jointplot(
        data=dataframe,
        x=x_col,
        y=y_col,
        kind="scatter",
        height=8,
        alpha=0.4,
    )
    plt.axline([1, 1], [10, 10], label="1:1", linestyle="--", color="black")

    # Set log scales if needed
    if log_x:
        g.ax_joint.set_xscale("log")
        g.ax_marg_x.set_xscale("log")

    if log_y:
        g.ax_joint.set_yscale("log")
        g.ax_marg_y.set_yscale("log")

    # Add KDE plots to the margins
    if log_x:
        sns.kdeplot(x=x_vals, ax=g.ax_marg_x, log_scale=True, color="blue", fill=True)
    else:
        sns.kdeplot(x=x_vals, ax=g.ax_marg_x, color="blue", fill=True)

    if log_y:
        sns.kdeplot(y=y_vals, ax=g.ax_marg_y, log_scale=True, color="blue", fill=True)
    else:
        sns.kdeplot(y=y_vals, ax=g.ax_marg_y, color="blue", fill=True)

    plt.suptitle(f"Joint Distribution of {x_col} and {y_col} (Original Scale)", y=1.02)
    plt.tight_layout()
    plt.show()

    # Plot KDEs by category if available
    if category_col is not None and "by_category" in kde_results:
        # Create a color map for categories
        categories = transform_info.get(
            "categories", list(kde_results["by_category"].keys())
        )
        n_categories = len(categories)

        if n_categories > 0:
            # Create a categorical plot first
            plt.figure(figsize=figsize)

            # Get a colormap with distinct colors
            category_cmap = plt.get_cmap(cmap, n_categories)
            colors = [category_cmap(i / n_categories) for i in range(n_categories)]

            # Create scatter plot by category
            for i, category in enumerate(categories):
                category_mask = dataframe[category_col] == category

                if category in kde_results["by_category"]:
                    cat_x = (
                        x_vals[category_mask]
                        if isinstance(x_vals, np.ndarray)
                        else np.array(x_vals[category_mask])
                    )
                    cat_y = (
                        y_vals[category_mask]
                        if isinstance(y_vals, np.ndarray)
                        else np.array(y_vals[category_mask])
                    )

                    plt.scatter(
                        cat_x, cat_y, alpha=0.5, s=20, color=colors[i], label=category
                    )

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"Data Points by {category_col}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # Plot KDE for each category
            plt.figure(figsize=figsize)

            # Plot the data points in background with low alpha
            plt.scatter(x_vals, y_vals, alpha=0.1, s=5, color="gray")

            # Plot contours for each category
            for i, category in enumerate(categories):
                if category in kde_results["by_category"]:
                    cat_kde = kde_results["by_category"][category]
                    Z = cat_kde(positions).reshape(X.shape)

                    # Plot the contour lines for this category
                    contour_lines = plt.contour(
                        X,
                        Y,
                        Z,
                        levels=int(n_levels / 2),
                        colors=[colors[i]],
                        linewidths=1.5,
                        alpha=0.8,
                        label=f"{category}",
                    )

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"Bivariate KDE Contours by {category_col}")

            # Create proxy artists for the legend
            legend_elements = [
                plt.Line2D([0], [0], color=colors[i], lw=2, label=cat)
                for i, cat in enumerate(categories)
                if cat in kde_results["by_category"]
            ]
            plt.legend(handles=legend_elements)

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # Create separate KDE plots for each category
            fig, axes = plt.subplots(int(np.ceil(n_categories / 2)), 2, figsize=figsize)
            axes = axes.flatten() if n_categories > 1 else [axes]

            for i, category in enumerate(categories):
                if i < len(axes) and category in kde_results["by_category"]:
                    ax = axes[i]

                    # Get category data
                    category_mask = dataframe[category_col] == category
                    cat_x = (
                        x_vals[category_mask]
                        if isinstance(x_vals, np.ndarray)
                        else np.array(x_vals[category_mask])
                    )
                    cat_y = (
                        y_vals[category_mask]
                        if isinstance(y_vals, np.ndarray)
                        else np.array(y_vals[category_mask])
                    )

                    # Plot data points
                    ax.scatter(cat_x, cat_y, alpha=0.4, s=10, color=colors[i])

                    # Evaluate and plot KDE
                    cat_kde = kde_results["by_category"][category]
                    Z = cat_kde(positions).reshape(X.shape)
                    contour = ax.contourf(
                        X,
                        Y,
                        Z,
                        levels=n_levels,
                        cmap=plt.get_cmap("viridis"),
                        alpha=0.3,
                    )
                    contour_lines = ax.contour(
                        X,
                        Y,
                        Z,
                        levels=int(n_levels / 2),
                        colors=["black"],
                        linewidths=0.5,
                    )

                    ax.set_title(f"{category} (n={np.sum(category_mask)})")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.grid(True, alpha=0.3)

            # Hide any unused axes
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)

            plt.suptitle(f"Individual KDEs by {category_col}", y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()


def progress_iter(iterable, verbose: bool = False, desc: str | None = None):
    """
    Return an iterator that shows a tqdm progress bar if:
    - verbose is True AND tqdm is available,
    otherwise return the original iterable.

    Parameters
    ----------
    iterable : any iterable
    verbose : bool
        If False, returns raw iterable. If True, try tqdm.
    desc : str | None
        Optional description for the progress bar.

    Examples
    --------
    for item in progress_iter(items, verbose=True, desc="Processing"):
        ...
    """
    if not verbose:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        # tqdm not installed or failed
        print(desc)
        return iterable
