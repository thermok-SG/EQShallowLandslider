#!/usr/bin/env python3
"""Compare transient GDP and equilibrium SINMAP-style wetness on Nepal terrain.

Both hydrology models receive the same prescribed recharge and feed the same
ShallowLandslider stability equation.  The diagnostic is intentionally not a
calibration: it isolates the consequence of groundwater storage and recession.
"""

from __future__ import annotations

import argparse
import json
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

from components.hydrology import create_wetness_coupler
from components.shallow_landslider import ShallowLandslider
from examples.plot_nepal_hydrologic_foundation import DEFAULT_DEM, _central_crop
from utils.utilities import apply_soil_depth, get_topo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--cols", type=int, default=400)
    parser.add_argument(
        "--full-dem",
        action="store_true",
        help="Use the complete DEM instead of the default central crop",
    )
    parser.add_argument("--max-soil-depth", type=float, default=1.5)
    parser.add_argument("--storm-recharge", type=float, default=5.0e-7)
    parser.add_argument("--storm-hours", type=float, default=24.0)
    parser.add_argument("--recovery-days", type=float, default=3.0)
    parser.add_argument("--hydraulic-conductivity", type=float, default=1.0e-5)
    parser.add_argument("--porosity", type=float, default=0.25)
    parser.add_argument("--cohesion", type=float, default=10_000.0)
    parser.add_argument("--friction-angle", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_output/nepal_gdp_sinmap_comparison.png"),
    )
    parser.add_argument(
        "--mechanism-output",
        type=Path,
        help="Optional mechanism/time-response figure path; derived from --output by default",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        help="Optional JSON summary path; derived from --output by default",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "rows",
        "cols",
        "max_soil_depth",
        "storm_hours",
        "hydraulic_conductivity",
        "porosity",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.storm_recharge < 0.0 or args.recovery_days < 0.0:
        raise ValueError("recharge and recovery duration cannot be negative")


def _evaluate(landslider: ShallowLandslider) -> tuple[np.ndarray, np.ndarray]:
    landslider.run_one_step()
    return (
        landslider.results["factor_of_safety"].copy(),
        landslider.results["unstable_mask"].copy(),
    )


def _map(ax, values, shape, title, *, cmap="Blues", vmin=None, vmax=None):
    image = ax.imshow(
        np.asarray(values).reshape(shape),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax=ax, shrink=0.78)


def _mechanism_figure(
    args: argparse.Namespace,
    *,
    gdp_storm_mean: float,
    sinmap_storm_mean: float,
    gdp_recovery_mean: float,
    sinmap_recovery_mean: float,
    gdp_unstable_count: int,
    sinmap_unstable_count: int,
    gdp_minimum_fos: float,
    sinmap_minimum_fos: float,
):
    """Plot the different model structures alongside their measured response."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")

    model_panels = (
        (
            axes[0, 0],
            "GDP: transient groundwater balance",
            "Recharge $R(t)$",
            "Stored saturated thickness $h_s(t)$\n"
            "$n\\,\\partial h_s/\\partial t = R - \\nabla\\cdot q$",
            "Lateral drainage + return flow",
            "Wetness $m(t)=h_s(t)/h_{soil}$\nMemory: yes",
            "#2166ac",
        ),
        (
            axes[0, 1],
            "SINMAP style: instantaneous equilibrium index",
            "Recharge $R$ + terrain $a,\\,\\beta$",
            "Topographic equilibrium\n$m=\\min[(R/T)(a/\\sin\\beta),1]$",
            "No storage-balance evolution",
            "Wetness changes with forcing immediately\nMemory: no",
            "#b2182b",
        ),
    )
    for axis, title, source, state, loss, output, color in model_panels:
        axis.set_title(title, fontsize=13, weight="bold")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        boxes = ((0.05, 0.72, source), (0.31, 0.40, state), (0.62, 0.10, output))
        for x, y, label in boxes:
            axis.text(
                x,
                y,
                label,
                transform=axis.transAxes,
                ha="left",
                va="center",
                fontsize=11,
                bbox={
                    "boxstyle": "round,pad=0.5",
                    "facecolor": color,
                    "edgecolor": color,
                    "alpha": 0.14,
                },
            )
        axis.annotate(
            "",
            xy=(0.42, 0.52),
            xytext=(0.23, 0.70),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 2},
        )
        axis.annotate(
            "",
            xy=(0.72, 0.22),
            xytext=(0.58, 0.39),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 2},
        )
        axis.text(0.05, 0.17, loss, transform=axis.transAxes, fontsize=10, color="0.3")

    time_axis = axes[1, 0]
    storm_end = args.storm_hours / 24.0
    recovery_end = storm_end + args.recovery_days
    gdp_times = [0.0, storm_end, recovery_end]
    gdp_values = [0.0, gdp_storm_mean, gdp_recovery_mean]
    sinmap_times = [0.0, 1.0e-6, storm_end, storm_end + 1.0e-6, recovery_end]
    sinmap_values = [0.0, sinmap_storm_mean, sinmap_storm_mean, 0.0, 0.0]
    time_axis.plot(gdp_times, gdp_values, "o-", color="#2166ac", linewidth=2.5, label="GDP")
    time_axis.plot(
        sinmap_times,
        sinmap_values,
        drawstyle="steps-post",
        color="#b2182b",
        linewidth=2.5,
        label="SINMAP style",
    )
    time_axis.axvspan(0.0, storm_end, color="0.75", alpha=0.25, label="Recharge on")
    time_axis.axvline(storm_end, color="0.4", linestyle="--", linewidth=1)
    time_axis.set(xlabel="Elapsed time (days)", ylabel="Mean core-node relative wetness")
    time_axis.set_ylim(-0.03, 1.05)
    time_axis.set_title("Observed response to the same forcing", weight="bold")
    time_axis.grid(alpha=0.2)
    time_axis.legend(loc="best")

    outcome_axis = axes[1, 1]
    models = ["GDP", "SINMAP style"]
    counts = [gdp_unstable_count, sinmap_unstable_count]
    colors = ["#2166ac", "#b2182b"]
    bars = outcome_axis.bar(models, counts, color=colors, alpha=0.8)
    outcome_axis.set_ylabel("Unstable core nodes after storm")
    outcome_axis.set_title(
        "Consequences for the unchanged stability model", weight="bold", pad=14
    )
    outcome_axis.set_ylim(0.0, max(counts) * 1.16 if max(counts) else 1.0)
    outcome_axis.grid(axis="y", alpha=0.2)
    for bar, count, minimum_fos in zip(
        bars, counts, (gdp_minimum_fos, sinmap_minimum_fos)
    ):
        outcome_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\nminimum FS={minimum_fos:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.suptitle(
        "Why GDP and SINMAP-style hydrology respond differently\n"
        "Identical prescribed recharge and ShallowLandslider stability equation",
        fontsize=16,
    )
    return fig


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if not args.dem.exists():
        raise FileNotFoundError(f"Nepal DEM not found: {args.dem}")

    full_grid, _, _ = get_topo(
        buffer=0.0,
        dem_type="SRTMGL1",
        load_dem=str(args.dem),
        grid_spacing=30.0,
    )
    grid = (
        full_grid
        if args.full_dem
        else _central_crop(full_grid, args.rows, args.cols)
    )
    apply_soil_depth(
        grid,
        max_soil_depth=args.max_soil_depth,
        distribution="elevation",
        relationship="linear",
    )
    bedrock = grid.add_zeros("bedrock__elevation", at="node")
    bedrock[:] = grid.at_node["topographic__elevation"] - grid.at_node["soil__depth"]

    router = PriorityFloodFlowRouter(
        grid,
        flow_metric="D8",
        depression_handler="fill",
        accumulate_flow=True,
    )
    router.run_one_step()
    gdp = create_wetness_coupler(
        grid,
        "gdp",
        hydraulic_conductivity=args.hydraulic_conductivity,
        porosity=args.porosity,
        initial_relative_wetness=0.0,
    )
    landslider = ShallowLandslider(
        grid,
        cohesion_eff=args.cohesion,
        angle_int_frict=args.friction_angle,
        wetness_source="field",
        selection_method="all",
        compute_displacement=False,
        split_by_width_config=None,
    )
    gdp.run_one_step(args.storm_hours * 3600.0, recharge_rate=args.storm_recharge)
    gdp_storm = gdp.relative_wetness.copy()
    gdp_fos, gdp_unstable = _evaluate(landslider)
    if args.recovery_days > 0.0:
        gdp.run_one_step(args.recovery_days * 86400.0, recharge_rate=0.0)
    gdp_recovery = gdp.relative_wetness.copy()

    soil = np.asarray(grid.at_node["soil__depth"], dtype=float)
    transmissivity = args.hydraulic_conductivity * soil
    sinmap = create_wetness_coupler(
        grid,
        "sinmap",
        transmissivity=transmissivity,
        recharge_rate=args.storm_recharge,
    )
    sinmap_storm = sinmap.relative_wetness.copy()
    sinmap_fos, sinmap_unstable = _evaluate(landslider)
    sinmap.run_one_step(1.0, recharge_rate=0.0)
    sinmap_recovery = sinmap.relative_wetness.copy()

    shape = grid.shape
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    _map(
        axes[0, 0], gdp_storm, shape, "GDP after storm", vmin=0.0, vmax=1.0
    )
    _map(
        axes[0, 1],
        sinmap_storm,
        shape,
        "SINMAP equilibrium during storm",
        vmin=0.0,
        vmax=1.0,
    )
    _map(
        axes[0, 2],
        gdp_storm - sinmap_storm,
        shape,
        "Wetness difference: GDP − SINMAP",
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
    )
    _map(
        axes[1, 0],
        gdp_recovery,
        shape,
        f"GDP after {args.recovery_days:g} d recovery",
        vmin=0.0,
        vmax=1.0,
    )
    _map(
        axes[1, 1],
        sinmap_recovery,
        shape,
        "SINMAP immediately after recharge stops",
        vmin=0.0,
        vmax=1.0,
    )
    unstable_difference = gdp_unstable.astype(int) - sinmap_unstable.astype(int)
    _map(
        axes[1, 2],
        unstable_difference,
        shape,
        "Storm instability: GDP only (+1), SINMAP only (−1)",
        cmap="RdBu",
        vmin=-1.0,
        vmax=1.0,
    )
    fig.suptitle(
        "GDP transient storage versus SINMAP-style equilibrium wetness\n"
        f"Same recharge={args.storm_recharge:g} m/s, K={args.hydraulic_conductivity:g} m/s, "
        "and ShallowLandslider stability equation",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    core = grid.core_nodes
    gdp_storm_mean = float(np.mean(gdp_storm[core]))
    sinmap_storm_mean = float(np.mean(sinmap_storm[core]))
    gdp_recovery_mean = float(np.mean(gdp_recovery[core]))
    sinmap_recovery_mean = float(np.mean(sinmap_recovery[core]))
    gdp_unstable_count = int(np.count_nonzero(gdp_unstable[core]))
    sinmap_unstable_count = int(np.count_nonzero(sinmap_unstable[core]))
    gdp_minimum_fos = float(np.nanmin(gdp_fos[core]))
    sinmap_minimum_fos = float(np.nanmin(sinmap_fos[core]))

    mechanism_output = args.mechanism_output or args.output.with_name(
        f"{args.output.stem}_mechanism{args.output.suffix}"
    )
    mechanism = _mechanism_figure(
        args,
        gdp_storm_mean=gdp_storm_mean,
        sinmap_storm_mean=sinmap_storm_mean,
        gdp_recovery_mean=gdp_recovery_mean,
        sinmap_recovery_mean=sinmap_recovery_mean,
        gdp_unstable_count=gdp_unstable_count,
        sinmap_unstable_count=sinmap_unstable_count,
        gdp_minimum_fos=gdp_minimum_fos,
        sinmap_minimum_fos=sinmap_minimum_fos,
    )
    mechanism_output.parent.mkdir(parents=True, exist_ok=True)
    mechanism.savefig(mechanism_output, dpi=180)
    plt.close(mechanism)

    results_output = args.results_output or args.output.with_name(
        f"{args.output.stem}_results.json"
    )
    results = {
        "grid_shape": list(shape),
        "core_node_count": int(core.size),
        "storm_recharge_m_per_s": args.storm_recharge,
        "storm_hours": args.storm_hours,
        "recovery_days": args.recovery_days,
        "hydraulic_conductivity_m_per_s": args.hydraulic_conductivity,
        "porosity": args.porosity,
        "sinmap_transmissivity": "hydraulic_conductivity * local_soil_depth",
        "sinmap_specific_contributing_area": "drainage_area / grid.dx",
        "gdp": {
            "mean_storm_wetness": gdp_storm_mean,
            "mean_recovery_wetness": gdp_recovery_mean,
            "unstable_storm_core_nodes": gdp_unstable_count,
            "minimum_storm_factor_of_safety": gdp_minimum_fos,
        },
        "sinmap": {
            "mean_storm_wetness": sinmap_storm_mean,
            "mean_recovery_wetness": sinmap_recovery_mean,
            "unstable_storm_core_nodes": sinmap_unstable_count,
            "minimum_storm_factor_of_safety": sinmap_minimum_fos,
        },
    }
    results_output.parent.mkdir(parents=True, exist_ok=True)
    results_output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Grid: {shape[0]} rows x {shape[1]} columns")
    print(
        "Mean storm wetness on core nodes: "
        f"GDP={gdp_storm_mean:.4f}, SINMAP={sinmap_storm_mean:.4f}"
    )
    print(
        "Mean recovery wetness on core nodes: "
        f"GDP={gdp_recovery_mean:.4f}, SINMAP={sinmap_recovery_mean:.4f}"
    )
    print(
        "Unstable storm core nodes: "
        f"GDP={gdp_unstable_count}, SINMAP={sinmap_unstable_count}"
    )
    print(
        "Minimum storm factor of safety: "
        f"GDP={gdp_minimum_fos:.4f}, SINMAP={sinmap_minimum_fos:.4f}"
    )
    print(f"Saved {args.output}")
    print(f"Saved {mechanism_output}")
    print(f"Saved {results_output}")


if __name__ == "__main__":
    main()
