#!/usr/bin/env python3
"""Create tabular, distribution, statistical, and spatial run diagnostics.

Run directories are discovered recursively beneath the supplied root. Every
run needs the v1.2 manifest, summary, region table, and raster bundle written by
the model CLI. Measured CSVs are optional: without them the command still
creates model summaries and maps; with them it overlays observed distributions
and calculates KS, Kuiper, and Wasserstein comparisons.
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis import (
    compare_run_distributions,
    discover_runs,
    load_observed_landslides,
    load_region_ensemble,
    load_run,
    plot_run,
    plot_run_maps,
    plot_parameter_sensitivity,
    summarize_run_distributions,
    swept_parameters,
)

_WORKER_OUTPUT_DIR = None
_WORKER_SELECTED_ONLY = True
_WORKER_OBSERVED = None


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _initialize_worker(output_dir, selected_only, observed):
    """Set process-local inputs once instead of serializing them for every run."""
    global _WORKER_OUTPUT_DIR, _WORKER_SELECTED_ONLY, _WORKER_OBSERVED
    _WORKER_OUTPUT_DIR = Path(output_dir)
    _WORKER_SELECTED_ONLY = selected_only
    _WORKER_OBSERVED = observed


def _analyse_run(run_dir):
    """Create all independent products for one run in a worker process."""
    run_dir = Path(run_dir)
    run = load_run(run_dir, load_rasters=True)
    try:
        distribution_figure = plot_run(
            run,
            selected_only=_WORKER_SELECTED_ONLY,
            observed=_WORKER_OBSERVED,
            output_path=_WORKER_OUTPUT_DIR / f"{run_dir.name}.png",
        )
        plt.close(distribution_figure)
        map_figure = plot_run_maps(
            run,
            output_path=_WORKER_OUTPUT_DIR / f"{run_dir.name}_maps.png",
        )
        plt.close(map_figure)
        summary = summarize_run_distributions(
            run, observed=_WORKER_OBSERVED, selected_only=_WORKER_SELECTED_ONLY
        )
        comparison = None
        if _WORKER_OBSERVED is not None:
            comparison = compare_run_distributions(
                run, observed=_WORKER_OBSERVED, selected_only=_WORKER_SELECTED_ONLY
            )
        return summary, comparison
    finally:
        plt.close("all")
        # Explicitly close a lazily opened xarray/Zarr store before the process
        # is reused for another ensemble member.
        rasters = run.get("rasters")
        close = getattr(rasters, "close", None)
        if close is not None:
            close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Model-only synthetic analysis
  python analyse_landslide_outputs.py --runs runs/synthetic_stability --output analysis_output/synthetic

  # Nepal model/observation comparison
  python analyse_landslide_outputs.py --runs runs --output analysis_output/nepal \\
    --observed-inventory input_data/nepal/measuredLandslides_all.csv \\
    --observed-zonal-stats input_data/nepal/measuredLandslides_all_ZonalStats.csv \\
    --min-observed-area 900

For synthetic terrain, pass only --observed-inventory if Nepal geometry is a
useful reference. Do not interpret Nepal elevation/slope as synthetic validation.""",
    )
    parser.add_argument(
        "--runs", required=True, help="Root directory containing run folders"
    )
    parser.add_argument(
        "--output", default="analysis_output", help="Analysis output directory"
    )
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument(
        "--observed-inventory",
        help="Measured inventory CSV containing area, length, and width",
    )
    parser.add_argument(
        "--observed-zonal-stats",
        help="Measured zonal-statistics CSV containing area, slope, and elevation",
    )
    parser.add_argument(
        "--min-observed-area",
        type=float,
        help="Exclude measured landslides smaller than this area in m²",
    )
    parser.add_argument(
        "--vary",
        action="append",
        default=[],
        metavar="PARAMETER",
        help=(
            "Limit controlled ensemble comparisons to this swept parameter "
            "(repeatable). By default every swept parameter is analysed."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        help=(
            "Number of runs to plot and summarize concurrently (default: 1). "
            "Each worker loads one full raster bundle, so size memory accordingly."
        ),
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_only = not args.include_candidates
    run_dirs = discover_runs(args.runs)
    # Keep the tabular run data in memory for the ensemble and every parameter
    # comparison. Previously each parameter caused every region table to be
    # read again from disk.
    tabular_runs = [load_run(run_dir, load_rasters=False) for run_dir in run_dirs]
    ensemble = load_region_ensemble(
        args.runs, selected_only=selected_only, runs=tabular_runs
    )
    ensemble.to_csv(output_dir / "region_ensemble.csv", index=False)

    observed = None
    if args.observed_inventory or args.observed_zonal_stats:
        observed = load_observed_landslides(
            inventory_path=args.observed_inventory,
            zonal_stats_path=args.observed_zonal_stats,
            min_area=args.min_observed_area,
        )

    automatic_parameters = not args.vary
    parameters_to_compare = args.vary or swept_parameters(args.runs, runs=tabular_runs)
    for parameter in parameters_to_compare:
        comparison_dir = (
            output_dir / "parameter_comparisons" / parameter.replace(".", "_")
        )
        try:
            sensitivity = plot_parameter_sensitivity(
                args.runs,
                parameter,
                comparison_dir,
                selected_only=selected_only,
                observed=observed,
                runs=tabular_runs,
            )
        except ValueError as exc:
            if not automatic_parameters or "No controlled comparison" not in str(exc):
                raise
            print(f"Skipping {parameter}: {exc}")
            continue
        sensitivity.to_csv(
            output_dir / f"parameter_sensitivity_{parameter.replace('.', '_')}.csv",
            index=False,
        )

    del tabular_runs
    summaries = []
    comparisons = []
    if args.jobs == 1:
        _initialize_worker(output_dir, selected_only, observed)
        results = (_analyse_run(run_dir) for run_dir in run_dirs)
        for summary, comparison in results:
            summaries.append(summary)
            if comparison is not None:
                comparisons.append(comparison)
    else:
        print(f"Analysing {len(run_dirs)} runs with {args.jobs} workers")
        # Spawn is deterministic across Linux and macOS and avoids inheriting
        # Matplotlib state from the parent process.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(output_dir, selected_only, observed),
        ) as executor:
            futures = {
                executor.submit(_analyse_run, run_dir): index
                for index, run_dir in enumerate(run_dirs)
            }
            ordered_results = [None] * len(run_dirs)
            for future in as_completed(futures):
                ordered_results[futures[future]] = future.result()
            for summary, comparison in ordered_results:
                summaries.append(summary)
                if comparison is not None:
                    comparisons.append(comparison)
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            output_dir / "distribution_summary.csv", index=False
        )
    if comparisons:
        pd.concat(comparisons, ignore_index=True).to_csv(
            output_dir / "distribution_comparison.csv", index=False
        )


if __name__ == "__main__":
    main()
