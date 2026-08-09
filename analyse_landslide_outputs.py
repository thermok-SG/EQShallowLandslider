#!/usr/bin/env python3
"""Create tabular, distribution, statistical, and spatial run diagnostics.

Run directories are discovered recursively beneath the supplied root. Every
run needs the v1.2 manifest, summary, region table, and raster bundle written by
the model CLI. Measured CSVs are optional: without them the command still
creates model summaries and maps; with them it overlays observed distributions
and calculates KS, Kuiper, and Wasserstein comparisons.
"""

import argparse
from pathlib import Path

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


def parse_args():
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
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_only = not args.include_candidates
    ensemble = load_region_ensemble(args.runs, selected_only=selected_only)
    ensemble.to_csv(output_dir / "region_ensemble.csv", index=False)

    automatic_parameters = not args.vary
    parameters_to_compare = args.vary or swept_parameters(args.runs)
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

    observed = None
    if args.observed_inventory or args.observed_zonal_stats:
        observed = load_observed_landslides(
            inventory_path=args.observed_inventory,
            zonal_stats_path=args.observed_zonal_stats,
            min_area=args.min_observed_area,
        )

    summaries = []
    comparisons = []
    for run_dir in discover_runs(args.runs):
        run = load_run(run_dir, load_rasters=True)
        distribution_figure = plot_run(
            run,
            selected_only=selected_only,
            observed=observed,
            output_path=output_dir / f"{run_dir.name}.png",
        )
        plt.close(distribution_figure)
        map_figure = plot_run_maps(
            run,
            output_path=output_dir / f"{run_dir.name}_maps.png",
        )
        plt.close(map_figure)
        summaries.append(
            summarize_run_distributions(
                run, observed=observed, selected_only=selected_only
            )
        )
        if observed is not None:
            comparisons.append(
                compare_run_distributions(
                    run, observed=observed, selected_only=selected_only
                )
            )
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
