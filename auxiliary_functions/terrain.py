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

from .topographic_functions import calculate_terrain_attribute


# %% Getting topography from OpenTopography
def get_topo(
    buffer: float, north=28.25, south=28.23, east=85.18, west=85.15, dem_type="NASADEM"
):
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
        xy_axis_units="m",
    )
    grid.add_field("topographic__elevation", z_geog, at="node")

    print(params)
    print(props)

    return grid, z_geog


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


# %% Generate soil depth array
def apply_soil_depth(
    grid,
    elevation_field="topographic__elevation",
    soil_field="soil__depth",
    max_soil_depth=1.5,
    distribution="uniform",
    relationship="linear",
    plot=False,
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

            print(
                f"Local curvature std window={window}, mean slope={np.nanmean(core_local_b):.3f}"
            )

        else:
            raise ValueError(f"Unknown curvature relationship: {relationship}")

        print(f"Curvature-based soil depth applied with '{relationship}' relationship.")
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

            print(f"Threshold drainage area: {drainage_threshold:.2f}")
            print(f"Nodes above threshold: {np.sum(above_threshold)}")

        elif drainage_transform == "linear":
            # Original linear approach (for comparison)
            max_drainage_area = np.max(core_drainage_area)
            if max_drainage_area > 0:
                normalized_drainage = core_drainage_area / max_drainage_area
            else:
                normalized_drainage = np.ones_like(core_drainage_area) * 0.5
            print(f"Linear drainage area range: 0 to {max_drainage_area:.2f}")

        else:
            raise ValueError(f"Unknown drainage_transform: {drainage_transform}")

        # Apply soil depth
        soil_depth[core_nodes] = normalized_drainage * max_soil_depth

        print(
            f"Drainage area-based soil depth applied using '{drainage_transform}' transformation."
        )
        print(
            f"Soil depth range: {np.min(soil_depth[core_nodes]):.3f} to {np.max(soil_depth[core_nodes]):.3f} m"
        )

    elif distribution == "mean_elev_curv":
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


# %%% Helper functions to create plot of soil depth distribution
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


# %%% Helper functions to compare soil depth distributions
def compare_soil_relationships(
    grid,
    elevation_field="topographic__elevation",
    max_soil_depth=1.0,
    plot=True,
    plot_type="combined",
):
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
        ("linear", 1.0),
        ("exponential", 1.0),
        ("power", 0.5),
        ("power", 1.0),
        ("power", 2.0),
        ("logarithmic", 1.0),
        ("sigmoid", 1.0),
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
        if rel == "power":
            field_name = f"soil__{rel}_exp{exp}"
            display_name = f"{rel} (exp={exp})"
        else:
            field_name = f"soil__{rel}"
            display_name = rel

        # Apply soil depth
        soil_depth = apply_soil_depth(
            grid,
            elevation_field=elevation_field,
            soil_field=field_name,
            max_soil_depth=max_soil_depth,
            distribution="elevation",
            relationship=rel,
            exponent=exp,
            plot=False,
        )  # Don't plot individual ones

        results[display_name] = soil_depth[core_nodes]
        print()

    # Create plots if requested
    if plot:
        if plot_type == "combined":
            _create_combined_plot(core_elevation, results, elevation_field)
        elif plot_type == "individual":
            _create_individual_plots(
                grid, elevation_field, relationships, max_soil_depth
            )
        elif plot_type == "grid":
            _create_grid_plot(core_elevation, results, elevation_field)
        else:
            raise ValueError("plot_type must be 'combined', 'individual', or 'grid'")

    return results


def _create_combined_plot(core_elevation, results, elevation_field):
    """Create a single plot comparing all relationships."""
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 8), layout="constrained")

    # Color palette for different relationships
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, (name, soil_depths) in enumerate(results.items()):
        plt.scatter(
            core_elevation, soil_depths, alpha=0.7, s=20, color=colors[i], label=name
        )

    plt.xlabel("Elevation (m)")
    plt.ylabel("Soil Depth (m)")
    plt.title("Comparison of Soil Depth Relationships")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add summary statistics
    stats_text = (
        f"Elevation range: {core_elevation.min():.2f} - {core_elevation.max():.2f} m\n"
    )
    stats_text += f"Core nodes: {len(core_elevation)}"

    plt.text(
        0.02,
        0.15,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.show()


def _create_grid_plot(core_elevation, results, elevation_field):
    """Create a subplot grid showing all relationships."""

    n_plots = len(results)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(15, 5 * n_rows), layout="constrained"
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (name, soil_depths) in enumerate(results.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        ax.scatter(core_elevation, soil_depths, alpha=0.7, s=20, color="blue")
        ax.set_xlabel("Elevation (m)")
        ax.set_ylabel("Soil Depth (m)")
        ax.set_title(f"{name.capitalize()} Relationship")
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats_text = f"Range: {soil_depths.min():.2f} - {soil_depths.max():.2f} m\n"
        stats_text += f"Mean: {soil_depths.mean():.2f} m"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Hide unused subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    plt.show()


def _create_individual_plots(grid, elevation_field, relationships, max_soil_depth):
    """Create individual plots for each relationship."""
    for rel, exp in relationships:
        if rel == "power":
            field_name = f"soil__{rel}_exp{exp}"
            display_name = f"{rel} (exp={exp})"
        else:
            field_name = f"soil__{rel}"
            display_name = rel

        create_soil_depth_plot(grid, elevation_field, field_name, display_name)
