#!/usr/bin/env python3
"""Run trigger-independent shallow-landslide transport on the Nepal DEM.

This is an integration diagnostic, not a calibrated rainfall-runout model. It
prescribes a spatial relative-wetness field, evaluates the existing
ShallowLandslider stability and aspect/KDE geometry pipeline at zero PGA,
selects every final candidate, and supplies an explicit mobility distance to
the unchanged multiflow runout calculation.

The default run uses a central DEM crop. Pass ``--rows 468 --cols 648`` to run
the complete bundled Nepal raster.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from landlab.components import PriorityFloodFlowRouter

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis import plot_pipeline_stages, plot_run_maps
from components.shallow_landslider import ShallowLandslider
from examples.plot_nepal_hydrologic_foundation import (
    DEFAULT_DEM,
    DEFAULT_INVENTORY,
    DEFAULT_KDE_BUNDLE,
    DEFAULT_ZONAL_STATS,
    _central_crop,
    _storm_wetness,
)
from utils.utilities import (
    apply_soil_depth,
    calculate_terrain_attribute,
    get_topo,
    pickle_or_not_to_pickle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--rows", type=int, default=140)
    parser.add_argument("--cols", type=int, default=180)
    parser.add_argument("--background-wetness", type=float, default=0.10)
    parser.add_argument("--storm-peak-wetness", type=float, default=0.95)
    parser.add_argument("--cohesion", type=float, default=10_000.0)
    parser.add_argument("--friction-angle", type=float, default=30.0)
    parser.add_argument("--max-soil-depth", type=float, default=1.5)
    parser.add_argument(
        "--runout-distance",
        type=float,
        default=60.0,
        help="Prescribed maximum travel distance in metres",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_output/nepal_hydrologic_runout.png"),
    )
    parser.add_argument(
        "--pipeline-output",
        type=Path,
        default=Path("analysis_output/nepal_runout_pipeline_stages.png"),
    )
    parser.add_argument("--kde-bundle", type=Path, default=DEFAULT_KDE_BUNDLE)
    parser.add_argument(
        "--disable-kde",
        action="store_true",
        help="Retain aspect splitting but skip measured-width KDE splitting",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.runout_distance < 0.0:
        raise ValueError("runout distance cannot be negative")
    if not args.dem.exists():
        raise FileNotFoundError(f"Nepal DEM not found: {args.dem}")

    full_grid, _, _ = get_topo(
        buffer=0.0,
        dem_type="SRTMGL1",
        load_dem=str(args.dem),
        grid_spacing=30.0,
    )
    grid = _central_crop(full_grid, args.rows, args.cols)
    apply_soil_depth(
        grid,
        max_soil_depth=args.max_soil_depth,
        distribution="elevation",
        relationship="linear",
    )
    wetness = _storm_wetness(
        grid, args.background_wetness, args.storm_peak_wetness
    )
    grid.add_field("soil__relative_wetness", wetness, at="node")

    router = PriorityFloodFlowRouter(
        grid,
        flow_metric="D8",
        separate_hill_flow=True,
        hill_flow_metric="Quinn",
        depression_handler="fill",
        update_hill_depressions=True,
        accumulate_flow=True,
    )
    router.run_one_step()

    split_by_width_config = None
    if not args.disable_kde:
        kde_bundle = pickle_or_not_to_pickle(
            {
                "file1": str(DEFAULT_INVENTORY),
                "file2": str(DEFAULT_ZONAL_STATS),
            },
            pickle_path=str(args.kde_bundle),
            min_area=900,
        )
        split_by_width_config = {
            "kde_data": kde_bundle["kde_data"],
            "kde_transform": kde_bundle["kde_transform"],
            "width_threshold": 1.5,
            "max_iterations": 5,
            "min_region_size": 10,
            "convergence_threshold": 0.75,
        }

    elevation_before = grid.at_node["topographic__elevation"].copy()
    soil_before = grid.at_node["soil__depth"].copy()
    landslider = ShallowLandslider(
        grid,
        cohesion_eff=args.cohesion,
        angle_int_frict=args.friction_angle,
        wetness_source="field",
        selection_method="all",
        compute_displacement=False,
        enable_runout=True,
        update_soil=True,
        runout_source_mode="selected",
        runout_distance=args.runout_distance,
        evolve_landscape=True,
        random_seed=0,
        split_by_width_config=split_by_width_config,
    )
    landslider.run_one_step()

    result = landslider.results
    runout = result["runout"]
    erosion_total = float(np.sum(runout["erosion"]))
    deposition_total = float(np.sum(runout["deposition"]))
    mass_error = deposition_total - erosion_total
    if not np.isclose(erosion_total, deposition_total):
        raise RuntimeError(
            "Runout failed mass balance: "
            f"erosion={erosion_total}, deposition={deposition_total}"
        )
    np.testing.assert_allclose(
        grid.at_node["topographic__elevation"],
        grid.at_node["bedrock__elevation"] + grid.at_node["soil__depth"],
    )
    soil_changed_count = int(
        np.count_nonzero(
            ~np.isclose(grid.at_node["soil__depth"], soil_before, equal_nan=True)
        )
    )
    elevation_changed_count = int(
        np.count_nonzero(
            ~np.isclose(
                grid.at_node["topographic__elevation"],
                elevation_before,
                equal_nan=True,
            )
        )
    )

    planform_curvature = calculate_terrain_attribute(
        grid,
        "topographic__elevation",
        "planform_curvature",
        out_field="terrain__planform_curvature",
    )
    profile_curvature = calculate_terrain_attribute(
        grid,
        "topographic__elevation",
        "profile_curvature",
        out_field="terrain__profile_curvature",
    )
    shape = grid.shape
    selected = result["selected_labels"] > 0
    affected = selected | (runout["erosion"] > 0) | (runout["deposition"] > 0)
    analysis_run = {
        "manifest": {"grid": {"dx": grid.dx, "dy": grid.dy}},
        "summary": {
            "run_id": (
                "Nepal zero-PGA trigger-independent runout diagnostic; "
                f"distance={args.runout_distance:g} m"
            )
        },
        "rasters": {
            "topographic_elevation": grid.at_node[
                "topographic__elevation"
            ].reshape(shape),
            "planform_curvature": planform_curvature.reshape(shape),
            "profile_curvature": profile_curvature.reshape(shape),
            "soil_depth": grid.at_node["soil__depth"].reshape(shape),
            "relative_wetness": wetness.reshape(shape),
            "critical_relative_wetness": result[
                "critical_relative_wetness"
            ].reshape(shape),
            "horizontal_pga": grid.at_node[
                "earthquake__horizontal_pga"
            ].reshape(shape),
            "factor_of_safety": result["factor_of_safety"].reshape(shape),
            "unstable_mask": result["unstable_mask"].reshape(shape),
            "selected_footprint": selected.reshape(shape),
            "runout_erosion": runout["erosion"].reshape(shape),
            "runout_deposition": runout["deposition"].reshape(shape),
            "runout_soil_depth_change": runout["soil_depth_change"].reshape(
                shape
            ),
            "combined_affected_footprint": affected.reshape(shape),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_run_maps(analysis_run, output_path=args.output)
    plt.close(figure)
    pipeline_figure = plot_pipeline_stages(
        result,
        elevation_before.reshape(shape),
        dx=grid.dx,
        dy=grid.dy,
        output_path=args.pipeline_output,
    )
    plt.close(pipeline_figure)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.pipeline_output}")
    print(f"Grid: {shape[0]} x {shape[1]}")
    print(f"Unstable nodes: {np.count_nonzero(result['unstable_mask']):,}")
    print(f"Selected source nodes: {runout['source_nodes'].size:,}")
    print(f"Eroded soil-column total: {erosion_total:.6f} m")
    print(f"Deposited soil-column total: {deposition_total:.6f} m")
    print(f"Mass-balance error: {mass_error:.3e} m")
    print(
        "Nodes with changed soil depth: "
        f"{soil_changed_count:,}"
    )
    print(
        "Nodes with changed surface elevation: "
        f"{elevation_changed_count:,}"
    )
    print(f"Terrain dirty after event: {landslider.terrain_dirty}")


if __name__ == "__main__":
    main()
