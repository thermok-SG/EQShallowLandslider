#!/usr/bin/env python3
"""Demonstrate transient GDP wetness on the bundled Nepal DEM.

The default uses the complete DEM. It applies the repository's current
elevation-based diagnostic soil-depth rule, initializes a shallow perched water
table, advances one prescribed-recharge storm and a zero-recharge recession,
and evaluates ShallowLandslider at zero PGA for each state.

This is a coupling and performance diagnostic, not a calibrated Nepal
rainfall–recharge model. Recharge is deliberately prescribed in m/s; the next
hydrology phase will calculate it from rainfall, unsaturated storage, and ET.
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

from components.hydrology import GroundwaterWetnessCoupler
from components.shallow_landslider import ShallowLandslider
from examples.plot_nepal_hydrologic_foundation import DEFAULT_DEM, _central_crop
from utils.utilities import apply_soil_depth, get_topo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument(
        "--rows",
        type=int,
        help="Optional central-crop rows; omit with --cols for the full DEM",
    )
    parser.add_argument(
        "--cols",
        type=int,
        help="Optional central-crop columns; omit with --rows for the full DEM",
    )
    parser.add_argument("--max-soil-depth", type=float, default=1.5)
    parser.add_argument("--initial-wetness", type=float, default=0.10)
    parser.add_argument(
        "--storm-recharge",
        type=float,
        default=5.0e-7,
        help="Prescribed groundwater recharge during the storm (m/s)",
    )
    parser.add_argument("--storm-hours", type=float, default=24.0)
    parser.add_argument("--recovery-days", type=float, default=3.0)
    parser.add_argument("--hydraulic-conductivity", type=float, default=1.0e-5)
    parser.add_argument("--porosity", type=float, default=0.25)
    parser.add_argument("--cohesion", type=float, default=10_000.0)
    parser.add_argument("--friction-angle", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_output/nepal_gdp_coupling_full.png"),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if (args.rows is None) != (args.cols is None):
        raise ValueError("--rows and --cols must be supplied together")
    for name in ("max_soil_depth", "storm_hours", "hydraulic_conductivity", "porosity"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.recovery_days < 0.0 or args.storm_recharge < 0.0:
        raise ValueError("recovery duration and recharge cannot be negative")
    if not 0.0 <= args.initial_wetness <= 1.0:
        raise ValueError("--initial-wetness must lie within [0, 1]")


def _evaluate(landslider: ShallowLandslider) -> tuple[np.ndarray, np.ndarray]:
    landslider.run_one_step()
    return (
        landslider.results["factor_of_safety"].copy(),
        landslider.results["unstable_mask"].copy(),
    )


def _map(ax, values, shape, title, *, cmap="viridis", vmin=None, vmax=None):
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
    grid = full_grid
    if args.rows is not None:
        grid = _central_crop(full_grid, args.rows, args.cols)

    apply_soil_depth(
        grid,
        max_soil_depth=args.max_soil_depth,
        distribution="elevation",
        relationship="linear",
    )
    bedrock = grid.add_zeros("bedrock__elevation", at="node")
    bedrock[:] = grid.at_node["topographic__elevation"] - grid.at_node["soil__depth"]

    hydrology = GroundwaterWetnessCoupler(
        grid,
        hydraulic_conductivity=args.hydraulic_conductivity,
        porosity=args.porosity,
        initial_relative_wetness=args.initial_wetness,
    )
    router = PriorityFloodFlowRouter(
        grid,
        flow_metric="D8",
        separate_hill_flow=True,
        hill_flow_metric="Quinn",
        depression_handler="fill",
        update_hill_depressions=True,
        accumulate_flow=True,
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

    background = hydrology.relative_wetness.copy()
    _, unstable_background = _evaluate(landslider)

    hydrology.run_one_step(
        args.storm_hours * 3600.0,
        recharge_rate=args.storm_recharge,
    )
    storm = hydrology.relative_wetness.copy()
    storm_runoff = hydrology.surface_runoff_rate.copy()
    hydrology.update_runoff_field()
    router.run_one_step()
    storm_fos, unstable_storm = _evaluate(landslider)

    if args.recovery_days > 0.0:
        hydrology.run_one_step(args.recovery_days * 86400.0, recharge_rate=0.0)
    recovery = hydrology.relative_wetness.copy()

    shape = grid.shape
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), layout="constrained")
    _map(
        axes[0, 0],
        grid.at_node["topographic__elevation"],
        shape,
        "Elevation (m)",
        cmap="terrain",
    )
    _map(
        axes[0, 1],
        grid.at_node["soil__depth"],
        shape,
        "Soil depth used by GDP (m)",
        cmap="YlOrBr",
        vmin=0.0,
    )
    _map(
        axes[0, 2],
        background,
        shape,
        "Initial relative wetness",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    _map(
        axes[0, 3],
        storm,
        shape,
        f"After {args.storm_hours:g} h recharge",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    _map(
        axes[1, 0],
        recovery,
        shape,
        f"After {args.recovery_days:g} d recovery",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    _map(
        axes[1, 1],
        storm - background,
        shape,
        "Storm wetness change",
        cmap="RdBu",
    )
    _map(
        axes[1, 2],
        storm_runoff * 86400.0 * 1000.0,
        shape,
        "GDP saturation-excess runoff (mm/day)",
        cmap="PuBu",
        vmin=0.0,
    )
    newly_unstable = unstable_storm & ~unstable_background
    _map(
        axes[1, 3],
        newly_unstable.astype(float),
        shape,
        "Newly unstable after storm",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
    )
    fig.suptitle(
        "Nepal transient groundwater coupling — prescribed recharge, zero PGA\n"
        f"K={args.hydraulic_conductivity:g} m/s; porosity={args.porosity:g}; "
        f"recharge={args.storm_recharge:g} m/s",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    core = grid.core_nodes
    print(f"Grid: {shape[0]} rows x {shape[1]} columns")
    print(
        f"GDP adaptive substeps in final call: {hydrology.groundwater.number_of_substeps}"
    )
    print(
        "Core-node wetness mean: "
        f"initial={np.mean(background[core]):.4f}, "
        f"storm={np.mean(storm[core]):.4f}, "
        f"recovery={np.mean(recovery[core]):.4f}"
    )
    print(
        "Unstable core nodes: "
        f"initial={np.count_nonzero(unstable_background[core])}, "
        f"storm={np.count_nonzero(unstable_storm[core])}, "
        f"new={np.count_nonzero(newly_unstable[core])}"
    )
    print(f"Storm minimum factor of safety: {np.nanmin(storm_fos[core]):.4f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
