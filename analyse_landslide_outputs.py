#!/usr/bin/env python3
"""Create summary tables and distribution plots from v1.2 run outputs."""

import argparse
from pathlib import Path

from analysis import discover_runs, load_region_ensemble, load_run, plot_run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, help="Root directory containing run folders")
    parser.add_argument("--output", default="analysis_output", help="Analysis output directory")
    parser.add_argument("--include-candidates", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_only = not args.include_candidates
    ensemble = load_region_ensemble(args.runs, selected_only=selected_only)
    ensemble.to_csv(output_dir / "region_ensemble.csv", index=False)

    for run_dir in discover_runs(args.runs):
        run = load_run(run_dir)
        plot_run(
            run,
            selected_only=selected_only,
            output_path=output_dir / f"{run_dir.name}.png",
        )


if __name__ == "__main__":
    main()
