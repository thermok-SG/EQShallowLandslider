#!/usr/bin/env python3
"""Compare transient GDP and equilibrium SINMAP-style wetness on Nepal terrain.

Both hydrology models receive the same prescribed recharge and feed the same
ShallowLandslider stability equation.  The diagnostic is intentionally not a
calibration: it isolates the consequence of groundwater storage and recession.
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
    print(f"Grid: {shape[0]} rows x {shape[1]} columns")
    print(
        "Mean storm wetness on core nodes: "
        f"GDP={np.mean(gdp_storm[core]):.4f}, "
        f"SINMAP={np.mean(sinmap_storm[core]):.4f}"
    )
    print(
        "Mean recovery wetness on core nodes: "
        f"GDP={np.mean(gdp_recovery[core]):.4f}, "
        f"SINMAP={np.mean(sinmap_recovery[core]):.4f}"
    )
    print(
        "Unstable storm core nodes: "
        f"GDP={np.count_nonzero(gdp_unstable[core])}, "
        f"SINMAP={np.count_nonzero(sinmap_unstable[core])}"
    )
    print(
        "Minimum storm factor of safety: "
        f"GDP={np.nanmin(gdp_fos[core]):.4f}, "
        f"SINMAP={np.nanmin(sinmap_fos[core]):.4f}"
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
