"""
Important utility functions to run ShallowLandslider separately
"""

# %% Import required packages
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
import json
import numpy as np
from scipy import stats
import richdem as rd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import pickle
from typing import Dict, Any
import logging
import platform
import subprocess
from logging.handlers import RotatingFileHandler

from landlab import RasterModelGrid, imshowhs_grid
from landlab.io import esri_ascii

from bmi_topography import Topography
from version import __version__


# %% Getting topography from OpenTopography


def apply_nodata_and_close_nodes(grid, z, nodata_values=(-9999,)):
    """Convert nodata to CLOSED nodes while keeping z finite."""
    z = z.astype("float32", copy=False)

    # Build mask: nodata values + existing NaNs
    nodata_mask = np.zeros_like(z, dtype=bool)
    for nd in nodata_values:
        nodata_mask |= z == nd
    nodata_mask |= np.isnan(z)

    # ---- FIX: modify live status_at_node, not a copy ----
    CLOSED = grid.BC_NODE_IS_CLOSED
    grid.status_at_node[nodata_mask] = CLOSED

    # Fill nodata values with a finite filler
    if (~nodata_mask).any():
        finite_min = np.nanmin(z[~nodata_mask])
    else:
        finite_min = 0.0

    fill_value = finite_min - 1.0
    z_finite = np.where(nodata_mask, fill_value, z).astype("float32", copy=False)

    # Save fields
    if "topographic__elevation" in grid.at_node:
        grid.at_node["topographic__elevation"][:] = z_finite
    else:
        grid.add_field("topographic__elevation", z_finite, at="node", copy=False)

    grid.add_field("nodata__mask", nodata_mask, at="node", copy=False)

    return z_finite, nodata_mask


def get_topo(
    buffer: float,
    north: float = 28.25,
    south: float = 28.23,
    east: float = 85.18,
    west: float = 85.15,
    dem_type: str = "NASADEM",
    smooth_num: int = 0,
    load_dem: str = None,
    verbose: bool = False,
    api_key: str = None,
):
    """
    Future-proof DEM loader that handles nodata correctly using CLOSED nodes.
    """

    try:
        # -------------------------------------------------------------
        # OPTION 1 — Load DEM from local file
        # -------------------------------------------------------------
        if load_dem is not None:
            if not Path(load_dem).exists():
                raise FileNotFoundError(f"DEM file not found: {load_dem}")

            with open(load_dem) as fp:
                grid_geog = esri_ascii.load(
                    fp, name="topographic__elevation", at="node"
                )

            z_geog = grid_geog.at_node["topographic__elevation"].astype("float32")

        else:
            # -------------------------------------------------------------
            # OPTION 2 — Download DEM from OpenTopo
            # -------------------------------------------------------------
            if north <= south or east <= west:
                raise ValueError("Invalid bounding box order.")

            params = Topography.DEFAULT.copy()
            params.update(
                {
                    "south": south - buffer,
                    "north": north + buffer,
                    "west": west - buffer,
                    "east": east + buffer,
                    "dem_type": dem_type,
                    "output_format": "AAIGrid",
                    "cache_dir": Path.cwd(),
                    "api_key": api_key,
                }
            )

            dem = Topography(**params)
            name = dem.fetch()
            props = dem.load()

            with open(name) as fp:
                grid_geog = esri_ascii.load(
                    fp, name="topographic__elevation", at="node"
                )

            z_geog = grid_geog.at_node["topographic__elevation"].astype("float32")

            if verbose:
                print("Request Parameters:", params)
                print("DEM Properties:", props)

        # -------------------------------------------------------------
        # Create Landlab RasterModelGrid with correct spacing
        # -------------------------------------------------------------
        spacing = {
            "SRTMGL3": 90,
            "COP90": 90,
            "SRTMGL1": 30,
            "AW3D30": 30,
            "NASADEM": 30,
            "COP30": 30,
        }.get(dem_type, 30)

        grid = RasterModelGrid(
            (grid_geog.number_of_node_rows, grid_geog.number_of_node_columns),
            xy_spacing=spacing,
            xy_axis_units="m",
        )

        # -------------------------------------------------------------
        # NODATA HANDLING (recommended approach)
        # -------------------------------------------------------------
        nodata_values = (-9999, 1e30, 3.4028235e38)  # typical ASCII nodata markers

        z_finite, nodata_mask = apply_nodata_and_close_nodes(
            grid, z_geog, nodata_values=nodata_values
        )

        # -------------------------------------------------------------
        # Optional smoothing (safe because nodata nodes are CLOSED)
        # -------------------------------------------------------------
        if smooth_num > 0:
            sm = smooth_elevation_grid(grid, method="gaussian", smooth_num=smooth_num)
            sm = sm.astype("float32", copy=False)
            grid.at_node["topographic__elevation"] = sm
            z_finite = sm

        grid.at_node["topographic__elevation"] = grid.at_node[
            "topographic__elevation"
        ].astype("float64", copy=False)

        return grid, z_finite, nodata_mask

    except Exception as e:
        raise RuntimeError(f"Error in get_topo: {e}")


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

        a = kwargs.get("a", max_soil_depth / 2.0)
        scale = kwargs.get("scale", 0.01)  # default scaling for Patton regression

        if relationship == "linear":
            # Plain linear
            b = kwargs.get("b", -1.0)
            soil_depth[core_nodes] = np.clip(
                a + b * core_kappa, 0.0, max_soil_depth
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
                h_vals, 0.0, max_soil_depth
            )

        elif relationship == "linear_std_global":
            # Patton-style global std regression
            curv_std = np.nanstd(core_kappa)
            b = (-446.3 * curv_std + 30.3) * scale
            soil_depth[core_nodes] = np.clip(
                a + b * core_kappa, 0.0, max_soil_depth
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
                a + core_local_b * core_kappa, 0.0, max_soil_depth
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
            relationship=relationship,
            plot=False,
            **kwargs,
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
        if "nodata__mask" in grid.at_node:
            valid_mask &= ~np.asarray(grid.at_node["nodata__mask"], dtype=bool)
        valid_mask &= grid.status_at_node != grid.BC_NODE_IS_CLOSED

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
    file_name_dict, pickle_path="measured_data.pkl", min_area=900, verbose: bool = False
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
    measured_data.head()

    # All measured landslide zonal statistics (elevation, slope, aspect)
    measured_spatial_stats = pd.read_csv(file_name_dict["file2"])
    measured_spatial_stats.head()

    # Filter out landslides below sensitivity threshold
    measured_spatial_stats_900greater = measured_spatial_stats.drop(
        measured_spatial_stats[measured_spatial_stats["Area_m2"] <= min_area].index
    )

    # Measured landslide zonal statistics inside selected area
    # measured_spatial_stats_clipped = pd.read_csv(file_name_dict["file3"])

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
        # "measured_spatial_stats_clipped": measured_spatial_stats_clipped,
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


def _json_safe(value):
    """Convert common NumPy/path values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _git_provenance():
    """Return best-effort Git provenance without making saving depend on Git."""
    repo = Path(__file__).resolve().parents[1]

    def run_git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    try:
        status = run_git("status", "--porcelain")
        commit = run_git("rev-parse", "HEAD")
        branch = run_git("branch", "--show-current")
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "commit_short": None,
            "branch": None,
            "tag": None,
            "dirty": None,
        }
    try:
        tag = run_git("describe", "--tags", "--exact-match") or None
    except subprocess.CalledProcessError:
        tag = None
    return {
        "commit": commit,
        "commit_short": commit[:7],
        "branch": branch,
        "tag": tag,
        "dirty": bool(status),
    }


def _software_versions():
    packages = ("landlab", "numpy", "pandas", "scipy", "scikit-image", "richdem")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _run_name(config):
    """Build a readable parameter name without using it as the run identity."""
    soil = config.get("soil_params", {})
    simulation = config.get("simulation", {})
    distribution = str(soil.get("distribution", "uniform"))
    if distribution not in {"uniform", "drainage_area"}:
        distribution += f"-{soil.get('relationship', 'linear')}"
    return "_".join(
        [
            "SL",
            f"c{int(soil.get('cohesion_eff', 15000))}",
            f"phi{int(soil.get('angle_int_frict', 30))}",
            f"sub{int(round(soil.get('submerged_soil_proportion', 0.5) * 100))}",
            distribution,
            str(simulation.get("selection_method", "probabilistic")),
            f"seed{config.get('random_seed', 5000)}",
        ]
    )


def _mean_by_label(values, labels, label_ids):
    values = np.asarray(values).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    valid = (labels > 0) & np.isfinite(values)
    size = int(labels.max()) + 1 if labels.size else 1
    sums = np.bincount(labels[valid], weights=values[valid], minlength=size)
    counts = np.bincount(labels[valid], minlength=size)
    means = np.full(size, np.nan, dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means[np.asarray(label_ids, dtype=int)]


def _max_by_label(values, labels, label_ids):
    values = np.asarray(values).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    size = int(labels.max()) + 1 if labels.size else 1
    valid = (labels > 0) & np.isfinite(values)
    maxima = np.full(size, -np.inf, dtype=float)
    np.maximum.at(maxima, labels[valid], values[valid])
    maxima[~np.isfinite(maxima)] = np.nan
    return maxima[np.asarray(label_ids, dtype=int)]


def _build_region_output(ls, config, run_id):
    """Create a label-preserving, analysis-ready region table."""
    results = ls.results
    regions = results["group_properties"].copy()
    if "label" not in regions.columns:
        index_name = regions.index.name or "index"
        regions = regions.reset_index().rename(columns={index_name: "label"})
    regions["label"] = regions["label"].astype(int)

    final_labels = results.get("split_labels")
    if final_labels is None:
        final_labels = results.get("aspect_labels")
    final_labels = np.asarray(final_labels, dtype=np.int64)
    label_ids = regions["label"].to_numpy()
    selected_labels = np.asarray(results["selected_labels"], dtype=np.int64)
    selected_ids = np.unique(selected_labels[selected_labels > 0])
    regions["selected"] = regions["label"].isin(selected_ids)

    counts = np.bincount(final_labels, minlength=int(final_labels.max()) + 1)
    regions["cell_count"] = counts[label_ids]
    grid = ls.grid
    regions["centroid_x"] = _mean_by_label(grid.node_x, final_labels, label_ids)
    regions["centroid_y"] = _mean_by_label(grid.node_y, final_labels, label_ids)
    regions["mean_fos"] = _mean_by_label(
        results["factor_of_safety"], final_labels, label_ids
    )
    regions["mean_critical_acceleration"] = _mean_by_label(
        results["a_transient"], final_labels, label_ids
    )
    regions["mean_a_diff"] = _mean_by_label(results["a_diff"], final_labels, label_ids)
    if results.get("newmark") is not None:
        regions["max_newmark_displacement"] = _max_by_label(
            results["newmark"], final_labels, label_ids
        )

    soil = config.get("soil_params", {})
    simulation = config.get("simulation", {})
    regions.insert(0, "run_id", run_id)
    regions["cohesion_eff"] = soil.get("cohesion_eff")
    regions["angle_int_frict"] = soil.get("angle_int_frict")
    regions["submerged_soil_proportion"] = soil.get("submerged_soil_proportion")
    regions["soil_distribution"] = soil.get("distribution")
    regions["soil_relationship"] = soil.get("relationship")
    regions["selection_method"] = simulation.get("selection_method")
    regions["random_seed"] = config.get("random_seed")
    return regions


def _collect_raster_arrays(ls):
    """Collect available node arrays under stable v1.2 output names."""
    results = ls.results
    arrays = {
        "factor_of_safety": results.get("factor_of_safety"),
        "critical_acceleration": results.get("a_transient"),
        "driving_acceleration": results.get("a_driving"),
        "driving_minus_critical_acceleration": results.get("a_diff"),
        "unstable_mask": results.get("unstable_mask"),
        "region_labels": results.get("labels"),
        "aspect_labels": results.get("aspect_labels"),
        "split_labels": results.get("split_labels"),
        "selected_labels": results.get("selected_labels"),
        "newmark_displacement": results.get("newmark"),
    }
    grid_fields = {
        "topographic_elevation": "topographic__elevation",
        "soil_depth": "soil__depth",
        "nodata_mask": "nodata__mask",
        "horizontal_pga": "earthquake__horizontal_pga",
        "vertical_pga": "earthquake__vertical_pga",
        "runout_erosion": "landslide__erosion",
        "runout_deposition": "landslide__deposition",
        "runout_soil_depth_change": "landslide__soil_depth_change",
    }
    for output_name, field_name in grid_fields.items():
        if field_name in ls.grid.at_node:
            arrays[output_name] = ls.grid.at_node[field_name]
    selected_labels = arrays.get("selected_labels")
    if selected_labels is not None:
        selected_footprint = np.asarray(selected_labels) > 0
        arrays["selected_footprint"] = selected_footprint
        if "runout_erosion" in arrays and "runout_deposition" in arrays:
            erosion_footprint = np.asarray(arrays["runout_erosion"]) > 0
            deposition_footprint = np.asarray(arrays["runout_deposition"]) > 0
            runout_footprint = erosion_footprint | deposition_footprint
            arrays.update(
                {
                    "runout_erosion_footprint": erosion_footprint,
                    "runout_deposition_footprint": deposition_footprint,
                    "runout_affected_footprint": runout_footprint,
                    "runout_only_footprint": runout_footprint & ~selected_footprint,
                    "combined_affected_footprint": runout_footprint
                    | selected_footprint,
                }
            )
    return {name: np.asarray(value) for name, value in arrays.items() if value is not None}


def _write_rasters(run_dir, ls, run_id, output_cfg, logger):
    arrays = _collect_raster_arrays(ls)
    shape = tuple(int(value) for value in ls.grid.shape)
    outputs = []
    write_zarr = bool(output_cfg.get("write_zarr", True))
    wrote_zarr = False

    if write_zarr:
        try:
            import xarray as xr
            import zarr  # noqa: F401

            data_vars = {
                name: (("y", "x"), values.reshape(shape))
                for name, values in arrays.items()
            }
            x = ls.grid.node_x.reshape(shape)[0, :]
            y = ls.grid.node_y.reshape(shape)[:, 0]
            dataset = xr.Dataset(data_vars=data_vars, coords={"x": x, "y": y})
            dataset.attrs.update(
                {"model_version": __version__, "run_id": run_id, "schema_version": 1}
            )
            chunk_cfg = output_cfg.get("zarr_chunks", [1024, 1024])
            chunks = (min(int(chunk_cfg[0]), shape[0]), min(int(chunk_cfg[1]), shape[1]))
            encoding = {name: {"chunks": chunks} for name in data_vars}
            zarr_path = run_dir / "rasters.zarr"
            dataset.to_zarr(
                zarr_path, mode="w", encoding=encoding, zarr_format=3
            )
            outputs.append("rasters.zarr")
            wrote_zarr = True
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("Zarr output unavailable (%s); using .npy rasters.", exc)

    if not wrote_zarr and output_cfg.get("write_npy_fallback", True):
        raster_dir = run_dir / "rasters"
        raster_dir.mkdir(exist_ok=True)
        for name, values in arrays.items():
            np.save(raster_dir / f"{name}.npy", values.reshape(shape), allow_pickle=False)
        with open(raster_dir / "metadata.json", "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "shape": list(shape),
                    "dx": float(ls.grid.dx),
                    "dy": float(ls.grid.dy),
                    "fields": sorted(arrays),
                },
                stream,
                indent=2,
            )
        outputs.append("rasters/")
    return outputs


def save_model_run(save_pickle, ls, config, output_dir, logger, runtime_metadata=None):
    """Save a versioned, self-describing ShallowLandslider v1.2 run directory."""
    git_provenance = _git_provenance()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc)
    parameter_name = _run_name(config)
    run_id = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{parameter_name}"
    run_dir = output_root / run_id
    run_dir.mkdir()
    output_cfg = config.get("outputs", {})

    regions = _build_region_output(ls, config, run_id)
    regions.to_csv(run_dir / "regions.csv", index=False)
    output_files = ["regions.csv"]
    if output_cfg.get("write_parquet", True):
        try:
            regions.to_parquet(run_dir / "regions.parquet", index=False)
            output_files.append("regions.parquet")
        except (ImportError, ModuleNotFoundError) as exc:
            logger.warning("Parquet output unavailable (%s); regions.csv was saved.", exc)

    selected = regions[regions["selected"]]
    valid_nodes = int(np.sum(ls.grid.status_at_node != ls.grid.BC_NODE_IS_CLOSED))
    selected_nodes = int(np.sum(np.asarray(ls.results["selected_labels"]) > 0))
    node_area_m2 = float(ls.grid.dx * ls.grid.dy)
    selected_footprint_percent = (
        100.0 * selected_nodes / valid_nodes
    ) if valid_nodes else 0.0
    summary = {
        "run_id": run_id,
        "model_version": __version__,
        "candidate_region_count": int(len(regions)),
        "selected_region_count": int(len(selected)),
        "selected_node_count": selected_nodes,
        "selected_footprint_node_count": selected_nodes,
        "selected_footprint_area_m2": selected_nodes * node_area_m2,
        "valid_node_count": valid_nodes,
        "selected_footprint_percent": selected_footprint_percent,
        # Backward-compatible v1.2 alias. Prefer selected_footprint_percent,
        # which states explicitly that this is not the runout footprint.
        "affected_node_percent": selected_footprint_percent,
        "selected_area_m2": float(selected["area"].sum()) if "area" in selected else 0.0,
        "selected_area_m2_median": (
            float(selected["area"].median()) if len(selected) and "area" in selected else None
        ),
        "selected_area_m2_max": (
            float(selected["area"].max()) if len(selected) and "area" in selected else None
        ),
        "selected_proportion": ls.results.get("selected_proportion"),
    }
    if "landslide__erosion" in ls.grid.at_node:
        erosion = np.asarray(ls.grid.at_node["landslide__erosion"])
        deposition = np.asarray(ls.grid.at_node["landslide__deposition"])
        runout_results = ls.results.get("runout") or {}
        failed_nodes = np.asarray(
            runout_results.get("failed_nodes", []), dtype=int
        )
        source_sums = np.asarray(
            list(runout_results.get("source_proportion_sums", {}).values()),
            dtype=float,
        )
        path_count_by_source = runout_results.get("source_path_counts", {})
        moving_sources = set(map(int, path_count_by_source)) & set(
            np.flatnonzero(erosion > 0).tolist()
        )
        source_path_counts = np.asarray(
            [
                count for source, count in path_count_by_source.items()
                if int(source) in moving_sources
            ],
            dtype=int,
        )
        selected_footprint = np.asarray(ls.results["selected_labels"]) > 0
        erosion_footprint = erosion > 0
        deposition_footprint = deposition > 0
        runout_footprint = erosion_footprint | deposition_footprint
        runout_only_footprint = runout_footprint & ~selected_footprint
        combined_footprint = runout_footprint | selected_footprint
        final_soil_depth = np.asarray(ls.grid.at_node["soil__depth"])
        summary.update(
            {
                "runout_enabled": True,
                "runout_changed_node_count": int(
                    np.count_nonzero(np.abs(deposition - erosion) > 0)
                ),
                "runout_source_node_count": int(failed_nodes.size),
                "runout_excavated_source_node_count": int(
                    np.count_nonzero(erosion > 0)
                ),
                "runout_traced_source_node_count": int(
                    len(runout_results.get("source_proportion_sums", {}))
                ),
                "runout_moving_source_node_count": int(len(moving_sources)),
                "runout_terminated_path_count": int(
                    len(runout_results.get("paths", []))
                ),
                "runout_mean_paths_per_moving_source": (
                    float(np.mean(source_path_counts))
                    if source_path_counts.size else 0.0
                ),
                "runout_max_paths_per_source": (
                    int(np.max(source_path_counts))
                    if source_path_counts.size else 0
                ),
                "runout_source_proportion_sum_min": (
                    float(np.min(source_sums)) if source_sums.size else None
                ),
                "runout_source_proportion_sum_max": (
                    float(np.max(source_sums)) if source_sums.size else None
                ),
                "runout_source_proportion_error_count": int(
                    np.count_nonzero(~np.isclose(source_sums, 1.0))
                ),
                "runout_erosion_footprint_node_count": int(
                    np.count_nonzero(erosion_footprint)
                ),
                "runout_deposition_footprint_node_count": int(
                    np.count_nonzero(deposition_footprint)
                ),
                "runout_affected_footprint_node_count": int(
                    np.count_nonzero(runout_footprint)
                ),
                "runout_affected_footprint_area_m2": float(
                    np.count_nonzero(runout_footprint) * node_area_m2
                ),
                "runout_affected_footprint_percent": (
                    100.0 * np.count_nonzero(runout_footprint) / valid_nodes
                ) if valid_nodes else 0.0,
                "selected_and_runout_overlap_node_count": int(
                    np.count_nonzero(runout_footprint & selected_footprint)
                ),
                "runout_only_footprint_node_count": int(
                    np.count_nonzero(runout_only_footprint)
                ),
                "runout_only_footprint_area_m2": float(
                    np.count_nonzero(runout_only_footprint) * node_area_m2
                ),
                "combined_affected_footprint_node_count": int(
                    np.count_nonzero(combined_footprint)
                ),
                "combined_affected_footprint_area_m2": float(
                    np.count_nonzero(combined_footprint) * node_area_m2
                ),
                "combined_affected_footprint_percent": (
                    100.0 * np.count_nonzero(combined_footprint) / valid_nodes
                ) if valid_nodes else 0.0,
                "runout_total_erosion_node_m": float(np.sum(erosion)),
                "runout_total_deposition_node_m": float(np.sum(deposition)),
                "runout_mass_balance_error_node_m": float(
                    np.sum(deposition) - np.sum(erosion)
                ),
                "final_soil_depth_min_m": float(np.nanmin(final_soil_depth)),
                "negative_final_soil_depth_node_count": int(
                    np.count_nonzero(final_soil_depth < 0)
                ),
            }
        )
    else:
        summary["runout_enabled"] = False
    with open(run_dir / "summary.json", "w", encoding="utf-8") as stream:
        json.dump(_json_safe(summary), stream, indent=2)
    output_files.append("summary.json")
    output_files.extend(_write_rasters(run_dir, ls, run_id, output_cfg, logger))

    if save_pickle:
        bundle = {
            "selected_group_props": regions,
            "grid_arrays": _collect_raster_arrays(ls),
            "config": config,
            "run_id": run_id,
            "model_version": __version__,
        }
        with open(run_dir / "run.pkl", "wb") as stream:
            pickle.dump(bundle, stream)
        output_files.append("run.pkl")

    dem_path = Path(config["dem_path"]) if config.get("dem_path") else None
    dem_info = {"path": str(dem_path) if dem_path else None}
    if dem_path and dem_path.exists():
        stat = dem_path.stat()
        dem_info.update(
            {
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    manifest = {
        "schema_version": 1,
        "model": {"name": "ShallowLandslider", "version": __version__},
        "run_id": run_id,
        "parameter_name": parameter_name,
        "created_utc": created_at.isoformat(),
        "git": git_provenance,
        "runtime": _json_safe(runtime_metadata or {}),
        "grid": {
            "shape": list(ls.grid.shape),
            "number_of_nodes": int(ls.grid.number_of_nodes),
            "dx": float(ls.grid.dx),
            "dy": float(ls.grid.dy),
            "xy_axis_units": str(getattr(ls.grid, "xy_axis_units", "unknown")),
            "crs": config.get("crs"),
        },
        "dem": dem_info,
        "config": _json_safe(config),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _software_versions(),
        },
        "outputs": output_files + ["manifest.json"],
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    logger.info("ShallowLandslider v%s outputs saved to %s", __version__, run_dir)
    return run_dir


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


# %% Plot results


def _ecdf(values):
    v = np.asarray(values)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    x = np.sort(v)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _first_existing_column(dataframe, candidates, label):
    """Return the first matching dataframe column from a list of aliases."""
    for column in candidates:
        if column in dataframe.columns:
            return column
    raise KeyError(
        f"Could not find {label} column. Tried {candidates}; "
        f"available columns are {list(dataframe.columns)}"
    )


def _clean_numeric(values, positive=False):
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if positive:
        values = values[values > 0]
    return values


def plot_comparison_panels_with_ecdf(
    observed_df,
    model_df,
    mg=None,
    labels_masked=None,
    cmap="terrain",
    title="Comparison Plots (Histograms + ECDF)",
    save_path=None,
):
    obs_area_col = _first_existing_column(
        observed_df, ("Area_m2", "area_m2", "Area", "area"), "observed area"
    )
    obs_elev_col = _first_existing_column(
        observed_df,
        ("Elevation_m_mean", "Elevation_mean", "mean_elev", "elevation", "median_elevation"),
        "observed elevation",
    )
    obs_slope_col = _first_existing_column(
        observed_df,
        ("Slope_deg_mean", "Mean_slope", "mean_slope", "slope", "median_slope"),
        "observed slope",
    )

    model_area = _clean_numeric(model_df["area"], positive=True)
    obs_area = _clean_numeric(observed_df[obs_area_col], positive=True)
    model_elev = _clean_numeric(model_df["median_elevation"])
    obs_elev = _clean_numeric(observed_df[obs_elev_col])
    model_slope = _clean_numeric(model_df["median_slope"])
    obs_slope = _clean_numeric(observed_df[obs_slope_col])

    if model_area.size == 0 or obs_area.size == 0:
        raise ValueError("Area comparison requires positive observed and model areas.")

    # Combine data for shared bins
    area_combined = np.concatenate([obs_area, model_area])
    elev_combined = np.concatenate([obs_elev, model_elev])
    slope_combined = np.concatenate([obs_slope, model_slope])

    # Use log-spaced bins for area
    bins_area = np.logspace(
        np.log10(area_combined.min()), np.log10(area_combined.max()), 20
    )
    bins_elev = np.histogram_bin_edges(elev_combined, bins=20)
    bins_slope = np.histogram_bin_edges(slope_combined, bins=20)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), layout="constrained")

    # Panel 1: Map (use solid red color)
    if mg is not None and labels_masked is not None:
        plt.sca(axes[0, 0])
        from matplotlib.colors import ListedColormap

        # Create a single-color colormap (solid red)
        single_red = ListedColormap(["red"])
        imshowhs_grid(
            mg,
            "topographic__elevation",
            plot_type="Drape1",
            drape1=labels_masked,
            cmap=single_red,
            allow_colorbar=False,
            cbar_or="vertical",
            ticks_km=True,
            cbar_loc="lower right",
            cbar_height=0.8,
            cbar_width=0.3,
        )
        axes[0, 0].set_title("Topographic Map")
    else:
        axes[0, 0].axis("off")
        axes[0, 0].text(0.5, 0.5, "Map not provided", ha="center", va="center")

    model_color = "tab:blue"
    obs_color = "tab:orange"

    # Panel 2: Area (log scale) - FIXED
    ax_area = axes[0, 1]
    sns.histplot(
        model_area,
        bins=bins_area,
        stat="density",
        color=model_color,
        alpha=0.6,
        label="Model",
        ax=ax_area,
    )
    sns.histplot(
        obs_area,
        bins=bins_area,
        stat="density",
        color=obs_color,
        alpha=0.6,
        label="Observed",
        ax=ax_area,
    )
    ax_area.set_xscale("log")  # Set log scale on x-axis
    ax_area.set_xlabel("Area [m²]")
    ax_area.set_title("Histogram of Area")
    ax_area.legend()

    ax_area_ecdf = ax_area.twinx()
    x_obs, y_obs = _ecdf(obs_area)
    x_mod, y_mod = _ecdf(model_area)
    if x_obs.size:
        ax_area_ecdf.plot(
            x_obs, y_obs, color="black", linestyle="--", label="Observed ECDF"
        )
    if x_mod.size:
        ax_area_ecdf.plot(x_mod, y_mod, color="gray", linestyle="-", label="Model ECDF")
    ax_area_ecdf.set_ylim(0, 1)
    ax_area_ecdf.set_ylabel("ECDF")

    # Panel 3: Elevation
    ax_elev = axes[1, 0]
    sns.histplot(
        model_elev,
        bins=bins_elev,
        stat="density",
        color=model_color,
        alpha=0.6,
        label="Model",
        ax=ax_elev,
    )
    sns.histplot(
        obs_elev,
        bins=bins_elev,
        stat="density",
        color=obs_color,
        alpha=0.6,
        label="Observed",
        ax=ax_elev,
    )
    ax_elev.set_xlabel("Elevation [m]")
    ax_elev.set_title("Histogram of Elevation")
    ax_elev.legend()

    ax_elev_ecdf = ax_elev.twinx()
    x_obs, y_obs = _ecdf(obs_elev)
    x_mod, y_mod = _ecdf(model_elev)
    if x_obs.size:
        ax_elev_ecdf.plot(x_obs, y_obs, color="black", linestyle="--")
    if x_mod.size:
        ax_elev_ecdf.plot(x_mod, y_mod, color="gray", linestyle="-")
    ax_elev_ecdf.set_ylim(0, 1)
    ax_elev_ecdf.set_ylabel("ECDF")

    # Panel 4: Slope
    ax_slope = axes[1, 1]
    sns.histplot(
        model_slope,
        bins=bins_slope,
        stat="density",
        color=model_color,
        alpha=0.6,
        label="Model",
        ax=ax_slope,
    )
    sns.histplot(
        obs_slope,
        bins=bins_slope,
        stat="density",
        color=obs_color,
        alpha=0.6,
        label="Observed",
        ax=ax_slope,
    )
    ax_slope.set_xlabel("Slope [degrees]")
    ax_slope.set_title("Histogram of Slope")
    ax_slope.legend()

    ax_slope_ecdf = ax_slope.twinx()
    x_obs, y_obs = _ecdf(obs_slope)
    x_mod, y_mod = _ecdf(model_slope)
    if x_obs.size:
        ax_slope_ecdf.plot(x_obs, y_obs, color="black", linestyle="--")
    if x_mod.size:
        ax_slope_ecdf.plot(x_mod, y_mod, color="gray", linestyle="-")
    ax_slope_ecdf.set_ylim(0, 1)
    ax_slope_ecdf.set_ylabel("ECDF")

    fig.suptitle(title, fontsize=16)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    plt.show()


# %% Output file preparation
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
    "angle_int_frict": "intfr",
}


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
            if (
                params["distribution"] == "curvature"
                and params["relationship"] == "linear"
            ):
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
        params["relationship"],  # can be None
        params["curvature_variant"],  # can be None
        params["random_seed"],  # can be None
    )


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


# %% Logging model progress
# utilities.py (or example_utils.py if you rename it)


def setup_logger(
    name: str = "landslider",
    log_dir: str = ".",
    log_file: str | None = None,
    level: str = "INFO",
    to_console: bool = True,
    rotate_mb: int = 50,
    backups: int = 3,
) -> logging.Logger:
    """
    Configure and return a logger for landslider workflows.

    Safe to call once per process. Subsequent calls return the same logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # prevent duplicate handlers

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    os.makedirs(log_dir, exist_ok=True)

    if log_file is None:
        log_file = os.path.join(log_dir, "run.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=rotate_mb * 1024 * 1024,
        backupCount=backups,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Optional console handler
    if to_console:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger
