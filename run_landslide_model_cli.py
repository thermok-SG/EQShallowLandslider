"""
Example script demonstrating how to run a standalone ShallowLandslider model from a YAML configuration file.

This script is not required to use the component itself.
"""

#!/usr/bin/env python3
import argparse
import os
import time
import yaml
import gc
import numpy as np

from shallow_landslide_component import ShallowLandslider
from utils import (
    get_topo,
    apply_soil_depth,
    calculate_terrain_attribute,
    generate_acceleration_grid,
    pickle_or_not_to_pickle,
    setup_logger,
    save_model_run
)
from scipy.ndimage import label as cc_label

from landlab.components import PriorityFloodFlowRouter
from landlab import RasterModelGrid


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


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Shallow Landslide Model via YAML config"
    )
    p.add_argument("--config", required=True, help="Path to YAML configuration file.")
    p.add_argument("--verbose_console", action="store_true")
    return p.parse_args()


def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def heartbeat(logger, msg, last_time, interval=300):
    now = time.time()
    if now - last_time > interval:
        logger.info(f"[HEARTBEAT] {msg} — still running...")
        return now
    return last_time


def main():
    args = parse_args()
    config = load_config(args.config)

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
    mg_full, z_full, _ = get_topo(
        dem_type=None,
        load_dem=dem_path,
        buffer=0,
        smooth_num=smooth_num,
    )

    z_full_2d = z_full.reshape(mg_full.shape)
    z_full_2d = z_full_2d.astype(float)
    
    nrows, ncols = z_full_2d.shape
    ncells = nrows * ncols
    logger.info(f"DEM size: {nrows} x {ncols} = {ncells:,} cells")
    last_beat = heartbeat(logger, "DEM Loaded", last_beat)

    # Determine chunking threshold
    chunk_cfg = config.get("chunking", {})
    chunk_threshold = int(chunk_cfg.get("threshold_cells", 20_000_000))
    tile_size = tuple(chunk_cfg.get("tile_size", (800, 800)))
    tile_overlap = int(chunk_cfg.get("overlap", 3))

    use_chunking = ncells >= chunk_threshold
    if use_chunking:
        logger.info(f"DEM exceeds threshold ({chunk_threshold:,}). Using chunked mode.")
    else:
        logger.info("DEM below threshold. Using full-grid mode.")

    split_cfg = config.get("split_by_width", {})
    kde_enabled = split_cfg.get("enabled", False)

    kde_data = None
    kde_transform = None

    if kde_enabled:
        kde_pkl = split_cfg.get("pickle_path")
        if kde_pkl is None:
            raise RuntimeError("KDE splitting enabled but no pickle_path provided.")

        # identical behaviour to original
        kde_bundle = pickle_or_not_to_pickle(
            file_name_dict={},  # unused for pickle mode
            pickle_path=kde_pkl,
            verbose=True,
        )

        kde_data = kde_bundle["kde_data"]
        kde_transform = kde_bundle["kde_transform"]
        
    soil_cfg = config.get("soil_params", {})
    eq_cfg = config.get("pga", {})

    if not use_chunking:
        logger.info("Running full-grid model...")

        # -----------------------------------
        # FLOW ROUTING
        # -----------------------------------
        flow_cfg = config.get("flow_router", {})
        if flow_cfg.get("enable", True):
            logger.info("Running PriorityFloodFlowRouter...")
            t1 = time.time()
            pf = PriorityFloodFlowRouter(
                mg_full,
                flow_metric=flow_cfg.get("flow_metric", "D8"),
                separate_hill_flow=flow_cfg.get("separate_hill_flow", True),
                depression_handler=flow_cfg.get("depression_handler", "fill"),
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
        apply_soil_depth(
            mg_full,
            max_soil_depth=soil_cfg.get("max_soil_depth", 1.5),
            distribution=soil_cfg.get("distribution", "uniform"),
            relationship=soil_cfg.get("relationship", "linear_std_local"),
            P0=soil_cfg.get("P0", 0.05),
            h_star=soil_cfg.get("h_star", 1.0),
            D=soil_cfg.get("D", 0.01),
            h_min=soil_cfg.get("h_min", 0.1),
            h_no_ss=soil_cfg.get("h_no_ss", 0.0),
            plot=False,
        )
        mg_full.add_zeros("bedrock__elevation", at="node", clobber=True)
        mg_full.at_node["bedrock__elevation"][:] = (
            mg_full.at_node["topographic__elevation"] - mg_full.at_node["soil__depth"]
        )

        # PGA
        pga_h, pga_v = generate_acceleration_grid(
            grid=mg_full,
            horizontal_max=eq_cfg.get("horizontal_max", 0.5),
            vertical_max=eq_cfg.get("vertical_max", 0.2),
            distribution=eq_cfg.get("distribution", "uniform"),
            plot_grids=False,
        )

        # Full-grid landslider
        ls = ShallowLandslider(
            mg_full,
            cohesion_eff=soil_cfg.get("cohesion_eff", 15e3),
            angle_int_frict=soil_cfg.get("angle_int_frict", 30),
            submerged_soil_proportion=soil_cfg.get("submerged_soil_proportion", 0.5),
            pga_h=pga_h,
            pga_v=pga_v,
            random_seed=seed,
            handle_small=config["simulation"].get("handle_small", "merge"),
            compute_displacement=config["simulation"].get(
                "compute_displacement", False
            ),
            enable_runout=config["simulation"].get("enable_runout", False),
            update_soil=config["simulation"].get("update_soil", False),
            verbose=config["simulation"].get("verbose", False),
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
            save_pickle=False,
            ls=ls, config=config,
            output_dir=config["output_dir"],
            logger=logger
            )
        

        logger.info("Results saved (full-grid mode) to csv")
        logger.info(f"=== FINISHED in {(time.time() - t0) / 3600:.2f} hours ===")
        return

    # ================================================================
    # CASE B — CHUNKED STABILITY COMPUTATION
    # ================================================================
    logger.info("Running chunked stability computation...")

    # Global arrays for tile-merged results
    global_fos = np.full_like(z_full_2d, np.nan, dtype=float)
    global_diff = np.full_like(z_full_2d, np.nan, dtype=float)
    global_unstable = np.zeros_like(z_full_2d, dtype=bool)

    # ---- TILE LOOP ----
    for r0, r1, c0, c1, re0, re1, ce0, ce1, tile_z in iterate_tiles(
        z_full_2d, tile_size=tile_size, overlap=tile_overlap
    ):
        logger.info(f"Tile r[{r0}:{r1}], c[{c0}:{c1}]")
        tile_z = tile_z.astype(float)

        # Create tile-local Landlab grid
        mg_tile = RasterModelGrid((re1 - re0, ce1 - ce0), xy_spacing=mg_full.dx)
        mg_tile.add_field("topographic__elevation", tile_z.ravel(), at="node")

        # Terrain attribute
        calculate_terrain_attribute(
            grid=mg_tile,
            field_name="topographic__elevation",
            attrib="planform_curvature",
        )

        # Soil
        if "soil__depth" not in mg_tile.at_node:
            mg_tile.add_zeros("soil__depth", at="node")
        apply_soil_depth(
            mg_tile,
            max_soil_depth=soil_cfg.get("max_soil_depth", 1.5),
            distribution=soil_cfg.get("distribution", "uniform"),
            relationship=soil_cfg.get("relationship", "linear_std_local"),
            P0=soil_cfg.get("P0", 0.05),
            h_star=soil_cfg.get("h_star", 1.0),
            D=soil_cfg.get("D", 0.01),
            h_min=soil_cfg.get("h_min", 0.1),
            h_no_ss=soil_cfg.get("h_no_ss", 0.0),
            plot=False,
        )
        mg_tile.add_zeros("bedrock__elevation", at="node", clobber=True)
        mg_tile.at_node["bedrock__elevation"][:] = (
            mg_tile.at_node["topographic__elevation"] - mg_tile.at_node["soil__depth"]
        )

        # PGA
        pga_h, pga_v = generate_acceleration_grid(
            grid=mg_tile,
            horizontal_max=eq_cfg.get("horizontal_max", 0.5),
            vertical_max=eq_cfg.get("vertical_max", 0.2),
            distribution=eq_cfg.get("distribution", "uniform"),
            plot_grids=False,
        )

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
        diff_tile = mg_tile.at_node[
            "landslide__driving_minus_critical_acceleration"
        ].reshape(tile_z.shape)
        unstable_tile = mg_tile.at_node["landslide__unstable_mask"].reshape(
            tile_z.shape
        )

        write_back_core(global_fos, fos_tile, r0, r1, c0, c1, re0, ce0)
        write_back_core(global_diff, diff_tile, r0, r1, c0, c1, re0, ce0)
        write_back_core(global_unstable, unstable_tile, r0, r1, c0, c1, re0, ce0)

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
    mg_global.add_field("landslide__factor_of_safety", global_fos.ravel(), at="node")
    mg_global.add_field(
        "landslide__driving_minus_critical_acceleration", global_diff.ravel(), at="node"
    )
    mg_global.add_field("landslide__unstable_mask", global_unstable.ravel(), at="node")

    # Soil field must be recomputed globally or stitched. We recompute for simplicity:
    mg_global.add_zeros("soil__depth", at="node")
    apply_soil_depth(
        mg_global,
        max_soil_depth=soil_cfg.get("max_soil_depth", 1.5),
        distribution=soil_cfg.get("distribution", "uniform"),
        relationship=soil_cfg.get("relationship", "linear_std_local"),
        P0=soil_cfg.get("P0", 0.05),
        h_star=soil_cfg.get("h_star", 1.0),
        D=soil_cfg.get("D", 0.01),
        h_min=soil_cfg.get("h_min", 0.1),
        h_no_ss=soil_cfg.get("h_no_ss", 0.0),
        plot=False,
    )
    mg_global.add_zeros("bedrock__elevation", at="node", clobber=True)
    mg_global.at_node["bedrock__elevation"][:] = (
        mg_global.at_node["topographic__elevation"] - mg_global.at_node["soil__depth"]
    )

    # PGA globally
    pga_h, pga_v = generate_acceleration_grid(
        grid=mg_global,
        horizontal_max=eq_cfg.get("horizontal_max", 0.5),
        vertical_max=eq_cfg.get("vertical_max", 0.2),
        distribution=eq_cfg.get("distribution", "uniform"),
        plot_grids=False,
    )

    # Now run the full global pipeline (from region identification onward)
    logger.info(
        "Running global aspect splitting, KDE splitting, selection, displacement..."
    )

    ls_global = ShallowLandslider(
        mg_global,
        cohesion_eff=soil_cfg.get("cohesion_eff", 15e3),
        angle_int_frict=soil_cfg.get.get("angle_int_frict", 30),
        submerged_soil_proportion=soil_cfg.get.get(
            "submerged_soil_proportion", 0.5
        ),
        pga_h=pga_h,
        pga_v=pga_v,
        random_seed=seed,
        handle_small=config["simulation"].get("handle_small", "merge"),
        compute_displacement=config["simulation"].get(
            "compute_displacement", False
        ),
        verbose=config["simulation"].get("verbose", False),
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
            save_pickle=False,
            ls=ls_global, config=config,
            output_dir=config["output_dir"],
            logger=logger
            )
        

    logger.info("Results saved (chunked global mode) to csv")
    logger.info(f"=== FINISHED in {(time.time() - t0) / 3600:.2f} hours ===")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()