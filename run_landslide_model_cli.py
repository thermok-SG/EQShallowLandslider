"""
Example script demonstrating how to run a standalone ShallowLandslider model from a YAML configuration file.

This script is not required to use the component itself.
"""

#!/usr/bin/env python3
import argparse
import copy
import gc
import os
import time

import numpy as np
import yaml
from landlab import RasterModelGrid
from landlab.components import PriorityFloodFlowRouter
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label as cc_label

from components.shallow_landslider import ShallowLandslider
from utils import (
    apply_soil_depth,
    generate_acceleration_grid,
    get_topo,
    pickle_or_not_to_pickle,
    save_model_run,
    setup_logger,
)


# ---------------------------------------------------------------------
# Utility: iterate tiles with overlap
# ---------------------------------------------------------------------
def iterate_tiles(array, tile_size=(800, 800), overlap=3):
    """
    Yield tiles of the DEM with overlap.

    Parameters
    ----------
    array : 2D ndarray
    tile_size : (int, int)
    overlap : int

    Yields
    ------
    (r0, r1, c0, c1, re0, re1, ce0, ce1, tile)
    """
    nrows, ncols = array.shape
    trows, tcols = tile_size

    for r0 in range(0, nrows, trows):
        for c0 in range(0, ncols, tcols):
            r1 = min(r0 + trows, nrows)
            c1 = min(c0 + tcols, ncols)

            # expanded region
            re0 = max(0, r0 - overlap)
            ce0 = max(0, c0 - overlap)
            re1 = min(nrows, r1 + overlap)
            ce1 = min(ncols, c1 + overlap)

            tile = array[re0:re1, ce0:ce1]
            yield (r0, r1, c0, c1, re0, re1, ce0, ce1, tile)


# ---------------------------------------------------------------------
# Utility: write back only the core portion of tile results
# ---------------------------------------------------------------------
def write_back_core(global_arr, tile_arr, r0, r1, c0, c1, re0, ce0):
    core = tile_arr[(r0 - re0) : (r1 - re0), (c0 - ce0) : (c1 - ce0)]
    global_arr[r0:r1, c0:c1] = core


def fill_nodata_for_terrain(elevation, nodata_mask):
    """Fill nodata from the nearest valid cell for tile terrain calculations."""
    elevation = np.asarray(elevation, dtype=float)
    nodata_mask = np.asarray(nodata_mask, dtype=bool) | ~np.isfinite(elevation)
    if elevation.shape != nodata_mask.shape:
        raise ValueError("elevation and nodata_mask must have the same shape")
    if nodata_mask.all():
        return np.zeros_like(elevation), nodata_mask
    if not nodata_mask.any():
        return elevation.copy(), nodata_mask

    nearest_valid = distance_transform_edt(
        nodata_mask, return_distances=False, return_indices=True
    )
    filled = elevation[tuple(nearest_valid)]
    return filled, nodata_mask


def apply_configured_soil_depth(grid, soil_cfg):
    """Apply the configured soil model, including optional relationship settings."""
    params = dict(soil_cfg)
    params.setdefault("max_soil_depth", 1.5)
    params.setdefault("distribution", "uniform")
    params.setdefault("relationship", "linear_std_local")
    params.setdefault("P0", 0.05)
    params.setdefault("h_star", 1.0)
    params.setdefault("D", 0.01)
    params.setdefault("h_min", 0.1)
    params.setdefault("h_no_ss", 0.0)
    params["plot"] = False
    return apply_soil_depth(grid, **params)


def required_curvature_overlap(soil_cfg):
    """Return the tile halo needed by a curvature-based soil model."""
    if soil_cfg.get("distribution", "uniform") not in {
        "curvature",
        "mean_elev_curv",
    }:
        return 0

    # RichDEM curvature uses neighboring cells. The local-standard-deviation
    # relationship needs an additional half-window beyond that stencil.
    if soil_cfg.get("relationship", "linear_std_local") == "linear_std_local":
        window = int(soil_cfg.get("window", 5))
        if window < 1:
            raise ValueError("soil_params.window must be at least 1")
        return 1 + window // 2
    return 1


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Shallow Landslide Model via YAML config"
    )
    p.add_argument("--config", required=True, help="Path to YAML configuration file.")
    p.add_argument("--verbose_console", action="store_true")
    p.add_argument(
        "--chunking",
        choices=("auto", "always", "never"),
        help="Override chunking.mode from the YAML file.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the YAML configuration and exit without loading the DEM.",
    )
    return p.parse_args()


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping at its root")
    return config


def prepare_config(config, chunking_override=None):
    """Apply CLI defaults and validate supported option combinations."""
    config = copy.deepcopy(config)
    if "flow_params" not in config and "flow_router" in config:
        config["flow_params"] = copy.deepcopy(config["flow_router"])
    defaults = {
        "soil_params": {},
        "pga": {},
        "simulation": {},
        "flow_params": {},
        "chunking": {},
        "split_by_width": {},
        "outputs": {},
    }
    for section, default in defaults.items():
        value = config.setdefault(section, default)
        if not isinstance(value, dict):
            raise ValueError(f"{section} must be a YAML mapping")

    if not config.get("dem_path"):
        raise ValueError("dem_path is required")
    if not isinstance(config.get("smooth_num", 0), int) or config.get("smooth_num", 0) < 0:
        raise ValueError("smooth_num must be a non-negative integer")
    if not isinstance(config.get("random_seed", 5000), int):
        raise ValueError("random_seed must be an integer")

    chunk_cfg = config["chunking"]
    if chunking_override is not None:
        chunk_cfg["mode"] = chunking_override
    elif "mode" not in chunk_cfg:
        # Backward compatibility with the pre-v1.2 boolean option.
        chunk_cfg["mode"] = "auto" if chunk_cfg.get("enable_auto", True) else "never"
    if chunk_cfg["mode"] not in {"auto", "always", "never"}:
        raise ValueError("chunking.mode must be one of: auto, always, never")

    tile_size = chunk_cfg.get("tile_size", [800, 800])
    if (
        not isinstance(tile_size, (list, tuple))
        or len(tile_size) != 2
        or any(not isinstance(value, int) or value < 1 for value in tile_size)
    ):
        raise ValueError("chunking.tile_size must contain two positive integers")
    if int(chunk_cfg.get("threshold_cells", 20_000_000)) < 1:
        raise ValueError("chunking.threshold_cells must be at least 1")
    if int(chunk_cfg.get("overlap", 3)) < 0:
        raise ValueError("chunking.overlap cannot be negative")

    soil = config["soil_params"]
    distribution = soil.get("distribution", "uniform")
    if distribution not in {
        "uniform", "elevation", "curvature", "drainage_area", "mean_elev_curv"
    }:
        raise ValueError(f"Unsupported soil_params.distribution: {distribution}")
    if float(soil.get("max_soil_depth", 1.5)) <= 0:
        raise ValueError("soil_params.max_soil_depth must be positive")
    if float(soil.get("cohesion_eff", 15_000)) < 0:
        raise ValueError("soil_params.cohesion_eff cannot be negative")
    friction = float(soil.get("angle_int_frict", 30))
    if not 0 < friction < 90:
        raise ValueError("soil_params.angle_int_frict must be between 0 and 90 degrees")
    submerged = float(soil.get("submerged_soil_proportion", 0.5))
    if not 0 <= submerged <= 1:
        raise ValueError("soil_params.submerged_soil_proportion must be between 0 and 1")
    relationship = soil.get("relationship", "linear_std_local")
    if distribution == "elevation" and relationship not in {
        "linear", "exponential", "power", "sigmoid"
    }:
        raise ValueError(f"Unsupported elevation soil relationship: {relationship}")
    if distribution in {"curvature", "mean_elev_curv"} and relationship not in {
        "linear", "piecewise", "linear_std_global", "linear_std_local"
    }:
        raise ValueError(f"Unsupported curvature soil relationship: {relationship}")
    if soil.get("drainage_transform", "log") not in {
        "log", "sqrt", "power", "threshold", "linear"
    }:
        raise ValueError("Unsupported soil_params.drainage_transform")
    required_curvature_overlap(soil)

    pga = config["pga"]
    if pga.get("distribution", "uniform") not in {
        "uniform", "circular", "square", "diamond", "exponential"
    }:
        raise ValueError(
            "pga.distribution must be uniform, circular, square, diamond, or exponential"
        )
    for key in ("horizontal_max", "vertical_max"):
        if float(pga.get(key, 0.5 if key == "horizontal_max" else 0.2)) < 0:
            raise ValueError(f"pga.{key} cannot be negative")
    center = pga.get("center")
    if center is not None and (
        not isinstance(center, (list, tuple))
        or len(center) != 2
        or any(not isinstance(value, int) for value in center)
    ):
        raise ValueError("pga.center must be null or [row, column] integers")
    if pga.get("seed") is not None and not isinstance(pga["seed"], int):
        raise ValueError("pga.seed must be null or an integer")

    simulation = config["simulation"]
    if simulation.get("selection_method", "probabilistic") not in {
        "probabilistic", "pga_weighted"
    }:
        raise ValueError("simulation.selection_method must be probabilistic or pga_weighted")
    if simulation.get("proportion_method", "conservative") not in {
        "empirical", "statistical", "risk_profile", "adaptive", "conservative"
    }:
        raise ValueError("Unsupported simulation.proportion_method")
    custom = simulation.get("custom_proportion")
    if custom is not None and not 0 < float(custom) <= 1:
        raise ValueError("simulation.custom_proportion must be null or in (0, 1]")
    if simulation.get("handle_small", "merge") not in {"merge", "remove", "keep"}:
        raise ValueError("simulation.handle_small must be merge, remove, or keep")
    if int(simulation.get("aspect_interval", 20)) < 1:
        raise ValueError("simulation.aspect_interval must be positive")
    if int(simulation.get("n_jobs", 1)) < 1:
        raise ValueError("simulation.n_jobs must be positive")
    if float(simulation.get("time_shaking", 0)) < 0:
        raise ValueError("simulation.time_shaking cannot be negative")
    if float(simulation.get("displacement_threshold", 0)) < 0:
        raise ValueError("simulation.displacement_threshold cannot be negative")

    compute = bool(simulation.get("compute_displacement", False))
    runout = bool(simulation.get("enable_runout", False))
    update_soil = bool(simulation.get("update_soil", False))
    if runout and not (compute and update_soil):
        raise ValueError(
            "simulation.enable_runout requires compute_displacement and update_soil"
        )
    if update_soil and not runout:
        raise ValueError("simulation.update_soil is only supported with enable_runout")
    if runout:
        flow = config["flow_params"]
        if not flow.get("enable", True) or not flow.get("separate_hill_flow", True):
            raise ValueError(
                "Runout requires flow_params.enable and separate_hill_flow to be true"
            )
        if flow.get("hill_flow_metric", "Quinn") not in {
            "Quinn", "Freeman", "Holmgren", "Dinf"
        }:
            raise ValueError(
                "Runout requires a multiple-flow flow_params.hill_flow_metric "
                "such as Quinn"
            )

    chunks = config["outputs"].get("zarr_chunks", [1024, 1024])
    if (
        not isinstance(chunks, (list, tuple))
        or len(chunks) != 2
        or any(not isinstance(value, int) or value < 1 for value in chunks)
    ):
        raise ValueError("outputs.zarr_chunks must contain two positive integers")
    if str(config.get("log_level", "INFO")).upper() not in {
        "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    }:
        raise ValueError("log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")

    split = config["split_by_width"]
    if float(split.get("min_area", 900)) < 0:
        raise ValueError("split_by_width.min_area cannot be negative")
    if float(split.get("width_threshold", 1.5)) <= 0:
        raise ValueError("split_by_width.width_threshold must be positive")
    convergence = float(split.get("convergence_threshold", 0.75))
    if not 0 <= convergence <= 1:
        raise ValueError("split_by_width.convergence_threshold must be between 0 and 1")
    if int(split.get("min_region_size", 10)) < 1:
        raise ValueError("split_by_width.min_region_size must be positive")
    if int(split.get("max_iterations", 5)) < 1:
        raise ValueError("split_by_width.max_iterations must be positive")
    return config


def validate_execution_mode(config, use_chunking):
    """Reject features for which chunked execution has no valid implementation."""
    if not use_chunking:
        if (
            config["soil_params"].get("distribution") == "drainage_area"
            and not config["flow_params"].get("enable", True)
        ):
            raise ValueError("drainage_area soil depth requires flow_params.enable: true")
        return

    if config["simulation"].get("enable_runout", False):
        raise ValueError("Runout is not supported in chunked mode; use chunking.mode: never")
    if config["soil_params"].get("distribution") == "drainage_area":
        raise ValueError(
            "drainage_area soil depth is not supported in chunked mode because flow "
            "routing is a global operation"
        )


def configured_pga(grid, pga_cfg, default_seed):
    """Generate a configured PGA field once on the model's global grid."""
    center = pga_cfg.get("center")
    pga_seed = pga_cfg.get("seed")
    if pga_seed is None:
        pga_seed = default_seed
    pga_h, pga_v = generate_acceleration_grid(
        grid=grid,
        horizontal_max=pga_cfg.get("horizontal_max", 0.5),
        vertical_max=pga_cfg.get("vertical_max", 0.2),
        distribution=pga_cfg.get("distribution", "uniform"),
        center=tuple(center) if center is not None else None,
        random_center=pga_cfg.get("random_center", False),
        seed=pga_seed,
        plot_grids=False,
    )
    nodata = (
        grid.at_node["nodata__mask"] if "nodata__mask" in grid.at_node else None
    )
    if nodata is not None:
        pga_h[np.asarray(nodata, dtype=bool)] = np.nan
        pga_v[np.asarray(nodata, dtype=bool)] = np.nan
    return pga_h, pga_v


def _build_measured_file_dict(split_cfg):
    csv_cfg = split_cfg.get("csv_paths", {})
    if not csv_cfg:
        return {}

    file_name_dict = {
        "file1": csv_cfg.get("inventory_csv"),
        "file2": csv_cfg.get("zonal_stats_csv"),
    }
    missing = [key for key, value in file_name_dict.items() if not value]
    if missing:
        raise RuntimeError(
            "KDE CSV configuration is incomplete. "
            f"Missing values for: {missing}"
        )
    return file_name_dict


def heartbeat(logger, msg, last_time, interval=300):
    now = time.time()
    if now - last_time > interval:
        logger.info(f"[HEARTBEAT] {msg} — still running...")
        return now
    return last_time


def main():
    args = parse_args()
    config = prepare_config(load_config(args.config), args.chunking)

    if args.validate_only:
        print(f"Configuration is valid: {args.config}")
        return

    dem_path = config["dem_path"]
    out_dir = config.get("output_dir", "./output")
    seed = config.get("random_seed", 5000)
    smooth_num = config.get("smooth_num", 4)
    save_pickle = config.get("save_pickle", False)

    logger = setup_logger(
        name="landslider",
        log_dir=out_dir,
        level=config.get("log_level", "INFO"),
        to_console=args.verbose_console,
    )
    logger.info("=== Shallow Landslider Run Started ===")
    logger.info(f"Using config: {args.config}")

    t0 = time.time()
    last_beat = t0

    # -----------------------------------
    # DEM LOAD
    # -----------------------------------
    logger.info(f"Loading DEM: {dem_path}")
    mg_full, z_full, nodata_full = get_topo(
        dem_type=config.get("dem_type", "SRTMGL1"),
        load_dem=dem_path,
        buffer=0,
        smooth_num=smooth_num,
    )

    z_full_2d = z_full.reshape(mg_full.shape)
    z_full_2d = z_full_2d.astype(float)
    nodata_full_2d = np.asarray(nodata_full, dtype=bool).reshape(mg_full.shape)
    nodata_full_2d |= ~np.isfinite(z_full_2d)
    if nodata_full_2d.all():
        raise ValueError("DEM contains no valid elevation cells")
    nodata_fallback = np.min(z_full_2d[~nodata_full_2d])
    z_full_2d[nodata_full_2d] = nodata_fallback
    
    nrows, ncols = z_full_2d.shape
    ncells = nrows * ncols
    logger.info(f"DEM size: {nrows} x {ncols} = {ncells:,} cells")
    last_beat = heartbeat(logger, "DEM Loaded", last_beat)

    # Determine chunking threshold
    chunk_cfg = config.get("chunking", {})
    chunk_mode = chunk_cfg["mode"]
    chunk_threshold = int(chunk_cfg.get("threshold_cells", 20_000_000))
    tile_size = tuple(chunk_cfg.get("tile_size", (800, 800)))
    tile_overlap = int(chunk_cfg.get("overlap", 3))

    use_chunking = chunk_mode == "always" or (
        chunk_mode == "auto" and ncells >= chunk_threshold
    )
    validate_execution_mode(config, use_chunking)
    if use_chunking:
        logger.info(
            "Using chunked mode (mode=%s, threshold=%s cells).",
            chunk_mode,
            f"{chunk_threshold:,}",
        )
        if config["flow_params"].get("enable", True):
            logger.info(
                "Global flow routing is skipped in chunked mode; it is not required "
                "for the configured soil and simulation options."
            )
    else:
        logger.info("Using full-grid mode (chunking.mode=%s).", chunk_mode)

    split_cfg = config.get("split_by_width", {})
    kde_enabled = split_cfg.get("enabled", False)

    kde_data = None
    kde_transform = None

    if kde_enabled:
        kde_pkl = split_cfg.get("pickle_path")
        if kde_pkl is None:
            raise RuntimeError("KDE splitting enabled but no pickle_path provided.")

        file_name_dict = _build_measured_file_dict(split_cfg)
        if not os.path.exists(kde_pkl) and not file_name_dict:
            raise RuntimeError(
                "KDE splitting is enabled, but the configured pickle does not exist "
                f"({kde_pkl}) and no split_by_width.csv_paths were provided."
            )

        os.makedirs(os.path.dirname(kde_pkl) or ".", exist_ok=True)
        kde_bundle = pickle_or_not_to_pickle(
            file_name_dict=file_name_dict,
            pickle_path=kde_pkl,
            min_area=split_cfg.get("min_area", 900),
            verbose=True,
        )

        kde_data = kde_bundle["kde_data"]
        kde_transform = kde_bundle["kde_transform"]
        
    soil_cfg = config.get("soil_params", {})
    eq_cfg = config.get("pga", {})

    required_overlap = required_curvature_overlap(soil_cfg)
    if use_chunking and tile_overlap < required_overlap:
        logger.warning(
            "Increasing chunk overlap from %d to %d cells for the configured "
            "curvature soil relationship.",
            tile_overlap,
            required_overlap,
        )
        tile_overlap = required_overlap

    if not use_chunking:
        logger.info("Running full-grid model...")

        # -----------------------------------
        # FLOW ROUTING
        # -----------------------------------
        flow_cfg = config.get("flow_params", config.get("flow_router", {}))
        if flow_cfg.get("enable", True):
            logger.info("Running PriorityFloodFlowRouter...")
            t1 = time.time()
            pf = PriorityFloodFlowRouter(
                mg_full,
                flow_metric=flow_cfg.get("flow_metric", "D8"),
                separate_hill_flow=flow_cfg.get("separate_hill_flow", True),
                hill_flow_metric=flow_cfg.get("hill_flow_metric", "Quinn"),
                depression_handler=flow_cfg.get(
                    "depression_handler",
                    flow_cfg.get("depression_handling", "fill"),
                ),
                update_hill_depressions=flow_cfg.get("update_hill_depressions", True),
                accumulate_flow=flow_cfg.get("accumulate_flow", True),
            )
            pf.run_one_step()
            del pf
            gc.collect()
            logger.info(
                f"Flow routing finished in {(time.time() - t1) / 60:.2f} minutes"
            )
            last_beat = heartbeat(logger, "Flow routing", last_beat)
        else:
            logger.info("PriorityFloodFlowRouter skipped")

        # Soil
        if "soil__depth" not in mg_full.at_node:
            mg_full.add_zeros("soil__depth", at="node")
        apply_configured_soil_depth(mg_full, soil_cfg)
        mg_full.add_zeros("bedrock__elevation", at="node", clobber=True)
        mg_full.at_node["bedrock__elevation"][:] = (
            mg_full.at_node["topographic__elevation"] - mg_full.at_node["soil__depth"]
        )

        # PGA
        pga_h, pga_v = configured_pga(mg_full, eq_cfg, seed)

        # Full-grid landslider
        ls = ShallowLandslider(
            mg_full,
            cohesion_eff=soil_cfg.get("cohesion_eff", 15e3),
            angle_int_frict=soil_cfg.get("angle_int_frict", 30),
            submerged_soil_proportion=soil_cfg.get("submerged_soil_proportion", 0.5),
            pga_h=pga_h,
            pga_v=pga_v,
            aspect_interval=config["simulation"].get("aspect_interval", 20),
            selection_method=config["simulation"].get(
                "selection_method", "probabilistic"
            ),
            proportion_method=config["simulation"].get(
                "proportion_method", "conservative"
            ),
            custom_proportion=config["simulation"].get("custom_proportion"),
            random_seed=seed,
            handle_small=config["simulation"].get("handle_small", "merge"),
            compute_displacement=config["simulation"].get(
                "compute_displacement", False
            ),
            time_shaking=config["simulation"].get("time_shaking", 0.0),
            displacement_threshold=config["simulation"].get(
                "displacement_threshold", 0.0
            ),
            enable_runout=config["simulation"].get("enable_runout", False),
            update_soil=config["simulation"].get("update_soil", False),
            verbose=config["simulation"].get("verbose", False),
            n_jobs=config["simulation"].get("n_jobs", 1),
            split_by_width_config=(
                {
                    "kde_data": kde_data,
                    "kde_transform": kde_transform,
                    "width_threshold": split_cfg.get("width_threshold", 1.5),
                    "convergence_threshold": split_cfg.get(
                        "convergence_threshold", 0.75
                    ),
                    "min_region_size": split_cfg.get("min_region_size", 10),
                    "max_iterations": split_cfg.get("max_iterations", 5),
                }
                if kde_enabled
                else None
            ),
        )

        ls.run_one_step()
        save_model_run(
            save_pickle=save_pickle,
            ls=ls, config=config,
            output_dir=out_dir,
            logger=logger,
            runtime_metadata={
                "execution_mode": "full_grid",
                "elapsed_seconds": time.time() - t0,
                "cell_count": ncells,
            },
            )
        

        logger.info("Results saved (full-grid mode)")
        logger.info(f"=== FINISHED in {(time.time() - t0) / 3600:.2f} hours ===")
        return

    # ================================================================
    # CASE B — CHUNKED STABILITY COMPUTATION
    # ================================================================
    logger.info("Running chunked stability computation...")

    # Global arrays for tile-merged results
    global_fos = np.full_like(z_full_2d, np.nan, dtype=float)
    global_critical = np.full_like(z_full_2d, np.nan, dtype=float)
    global_driving = np.full_like(z_full_2d, np.nan, dtype=float)
    global_diff = np.full_like(z_full_2d, np.nan, dtype=float)
    global_unstable = np.zeros_like(z_full_2d, dtype=bool)
    # Landlab validates this required component input as Python float/float64.
    global_soil = np.zeros_like(z_full_2d, dtype=float)

    # PGA must be generated in global coordinates. Re-generating a non-uniform
    # distribution independently inside every tile changes its centre and scale.
    pga_h_full, pga_v_full = configured_pga(mg_full, eq_cfg, seed)
    pga_h_full_2d = pga_h_full.reshape(mg_full.shape)
    pga_v_full_2d = pga_v_full.reshape(mg_full.shape)

    # ---- TILE LOOP ----
    for r0, r1, c0, c1, re0, re1, ce0, ce1, tile_z in iterate_tiles(
        z_full_2d, tile_size=tile_size, overlap=tile_overlap
    ):
        logger.info(f"Tile r[{r0}:{r1}], c[{c0}:{c1}]")
        tile_mask = nodata_full_2d[re0:re1, ce0:ce1]
        tile_z, tile_mask = fill_nodata_for_terrain(tile_z, tile_mask)
        if tile_mask.all():
            logger.info("Skipping tile because it contains only nodata cells")
            continue
        write_back_core(z_full_2d, tile_z, r0, r1, c0, c1, re0, ce0)

        # Create tile-local Landlab grid
        mg_tile = RasterModelGrid((re1 - re0, ce1 - ce0), xy_spacing=mg_full.dx)
        mg_tile.add_field("topographic__elevation", tile_z.ravel(), at="node")
        mg_tile.add_field("nodata__mask", tile_mask.ravel(), at="node")
        mg_tile.status_at_node[tile_mask.ravel()] = mg_tile.BC_NODE_IS_CLOSED

        # Soil
        soil_tile = apply_configured_soil_depth(mg_tile, soil_cfg)
        soil_tile[tile_mask.ravel()] = 0.0
        mg_tile.add_zeros("bedrock__elevation", at="node", clobber=True)
        mg_tile.at_node["bedrock__elevation"][:] = (
            mg_tile.at_node["topographic__elevation"] - mg_tile.at_node["soil__depth"]
        )

        # PGA
        pga_h = pga_h_full_2d[re0:re1, ce0:ce1].ravel()
        pga_v = pga_v_full_2d[re0:re1, ce0:ce1].ravel()

        # ONLY stability computations
        ls_tile = ShallowLandslider(
            mg_tile,
            cohesion_eff=soil_cfg.get("cohesion_eff", 15e3),
            angle_int_frict=soil_cfg.get("angle_int_frict", 30),
            submerged_soil_proportion=soil_cfg.get(
                "submerged_soil_proportion", 0.5
            ),
            pga_h=pga_h,
            pga_v=pga_v,
            random_seed=seed,
            handle_small=config["simulation"].get("handle_small", "merge"),
            compute_displacement=False,  # do not do displacement in tiles
            verbose=False,
        )
        ls_tile._compute_stability()  # <<< stability only

        fos_tile = mg_tile.at_node["landslide__factor_of_safety"].reshape(tile_z.shape)
        critical_tile = ls_tile._a_transient.reshape(tile_z.shape)
        driving_tile = ls_tile._a_driving.reshape(tile_z.shape)
        diff_tile = mg_tile.at_node[
            "landslide__driving_minus_critical_acceleration"
        ].reshape(tile_z.shape)
        unstable_tile = mg_tile.at_node["landslide__unstable_mask"].reshape(
            tile_z.shape
        )
        fos_tile[tile_mask] = np.nan
        critical_tile[tile_mask] = np.nan
        driving_tile[tile_mask] = np.nan
        diff_tile[tile_mask] = np.nan
        unstable_tile[tile_mask] = False

        write_back_core(global_fos, fos_tile, r0, r1, c0, c1, re0, ce0)
        write_back_core(
            global_critical, critical_tile, r0, r1, c0, c1, re0, ce0
        )
        write_back_core(global_driving, driving_tile, r0, r1, c0, c1, re0, ce0)
        write_back_core(global_diff, diff_tile, r0, r1, c0, c1, re0, ce0)
        write_back_core(global_unstable, unstable_tile, r0, r1, c0, c1, re0, ce0)
        write_back_core(
            global_soil,
            soil_tile.reshape(tile_z.shape),
            r0,
            r1,
            c0,
            c1,
            re0,
            ce0,
        )

        del mg_tile, ls_tile
        gc.collect()

    logger.info("Finished tile stability. Merging global regions...")

    # ================================================================
    # STEP 3: Global connected-component region labeling
    # ================================================================
    global_labels, _ = cc_label(global_unstable, structure=np.ones((3, 3)))

    # ================================================================
    # STEP 4: Build global Landlab grid and assign fields to run splitting
    # ================================================================
    mg_global = RasterModelGrid((nrows, ncols), xy_spacing=mg_full.dx)
    mg_global.add_field("topographic__elevation", z_full_2d.ravel(), at="node")
    mg_global.add_field("nodata__mask", nodata_full_2d.ravel(), at="node")
    mg_global.status_at_node[nodata_full_2d.ravel()] = mg_global.BC_NODE_IS_CLOSED
    mg_global.add_field("landslide__factor_of_safety", global_fos.ravel(), at="node")
    mg_global.add_field(
        "landslide__critical_acceleration", global_critical.ravel(), at="node"
    )
    mg_global.add_field(
        "landslide__driving_minus_critical_acceleration", global_diff.ravel(), at="node"
    )
    mg_global.add_field("landslide__unstable_mask", global_unstable.ravel(), at="node")

    # Reuse the soil depths that produced the tile stability results. Recomputing
    # curvature here would allocate full-DEM RichDEM arrays and defeat chunking.
    mg_global.add_field(
        "soil__depth", global_soil.ravel(), at="node", copy=True
    )
    mg_global.add_zeros("bedrock__elevation", at="node", clobber=True)
    mg_global.at_node["bedrock__elevation"][:] = (
        mg_global.at_node["topographic__elevation"] - mg_global.at_node["soil__depth"]
    )

    # Reuse exactly the PGA fields used by the tiles.
    pga_h, pga_v = pga_h_full, pga_v_full

    # Now run the full global pipeline (from region identification onward)
    logger.info(
        "Running global aspect splitting, KDE splitting, selection, displacement..."
    )

    ls_global = ShallowLandslider(
        mg_global,
        cohesion_eff=soil_cfg.get("cohesion_eff", 15e3),
        angle_int_frict=soil_cfg.get("angle_int_frict", 30),
        submerged_soil_proportion=soil_cfg.get(
            "submerged_soil_proportion", 0.5
        ),
        pga_h=pga_h,
        pga_v=pga_v,
        aspect_interval=config["simulation"].get("aspect_interval", 20),
        selection_method=config["simulation"].get(
            "selection_method", "probabilistic"
        ),
        proportion_method=config["simulation"].get(
            "proportion_method", "conservative"
        ),
        custom_proportion=config["simulation"].get("custom_proportion"),
        random_seed=seed,
        handle_small=config["simulation"].get("handle_small", "merge"),
        compute_displacement=config["simulation"].get(
            "compute_displacement", False
        ),
        time_shaking=config["simulation"].get("time_shaking", 0.0),
        displacement_threshold=config["simulation"].get(
            "displacement_threshold", 0.0
        ),
        verbose=config["simulation"].get("verbose", False),
        n_jobs=config["simulation"].get("n_jobs", 1),
        split_by_width_config=(
            {
                "kde_data": kde_data,
                "kde_transform": kde_transform,
                "width_threshold": split_cfg.get("width_threshold", 1.5),
                "convergence_threshold": split_cfg.get("convergence_threshold", 0.75),
                "min_region_size": split_cfg.get("min_region_size", 10),
                "max_iterations": split_cfg.get("max_iterations", 5),
            }
            if kde_enabled
            else None
        ),
    )

    # Inject global region labels before pipeline
    ls_global._unstable_mask = global_unstable.ravel()
    ls_global._fos = global_fos.ravel()
    ls_global._a_transient = global_critical.ravel()
    ls_global._a_driving = global_driving.ravel()
    ls_global._a_diff = global_diff.ravel()
    ls_global.grid.at_node["landslide__unstable_mask"] = global_unstable.ravel()
    ls_global._labels = global_labels.ravel()
    ls_global.grid.at_node["landslide__region_labels"] = global_labels.ravel()

    # Continue from aspect-splitting onward
    ls_global._filter_by_aspect_and_split()
    ls_global._compute_group_properties()
    ls_global._select_groups()

    # Optional displacement
    if config["simulation"].get("compute_displacement", False):
        ls_global._compute_displacement(ls_global.time_shaking)
    
    save_model_run(
            save_pickle=save_pickle,
            ls=ls_global, config=config,
            output_dir=out_dir,
            logger=logger,
            runtime_metadata={
                "execution_mode": "chunked",
                "elapsed_seconds": time.time() - t0,
                "cell_count": ncells,
                "tile_size": list(tile_size),
                "tile_overlap": tile_overlap,
            },
            )
        

    logger.info("Results saved (chunked global mode)")
    logger.info(f"=== FINISHED in {(time.time() - t0) / 3600:.2f} hours ===")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
