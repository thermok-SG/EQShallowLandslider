#!/usr/bin/env python3
"""Plot the ShallowLandslider 2.0 wetness foundation on a Nepal DEM.

This example is an interface and stability demonstration, not yet a rainfall
or groundwater model. It loads the bundled Nepal SRTM DEM, extracts a central
subregion, constructs the elevation-based soil-depth distribution used by the
Nepal configuration, and prescribes two relative-wetness states:

* a spatially uniform background state; and
* a synthetic Gaussian storm footprint.

The same ``ShallowLandslider`` instance evaluates both states with zero PGA.
The resulting figure shows terrain, soil depth, critical relative wetness,
prescribed storm wetness, factor of safety, and unstable-node masks.

Run from the repository root:

    python examples/plot_nepal_hydrologic_foundation.py

The default output is ignored by Git:

    analysis_output/nepal_hydrologic_foundation.png

This script intentionally plots the raw physical instability mask rather than
selected landslides. Hydrologic probabilistic selection and landscape-evolving
runout are later ShallowLandslider 2.0 stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from landlab import RasterModelGrid

# Make the repository modules importable when this file is executed directly
# from ``examples/`` without requiring an editable package installation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from components.shallow_landslider import ShallowLandslider
from utils.utilities import apply_soil_depth, get_topo


DEFAULT_DEM = Path(
    "input_data/dem/"
    "SRTMGL1_28.169999999999998_85.03_28.3_85.21000000000001.asc"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    parser.add_argument("--rows", type=int, default=140)
    parser.add_argument("--cols", type=int, default=180)
    parser.add_argument("--row-offset", type=int, default=0)
    parser.add_argument("--col-offset", type=int, default=0)
    parser.add_argument("--background-wetness", type=float, default=0.10)
    parser.add_argument("--storm-peak-wetness", type=float, default=0.95)
    parser.add_argument("--cohesion", type=float, default=10_000.0)
    parser.add_argument("--friction-angle", type=float, default=30.0)
    parser.add_argument("--max-soil-depth", type=float, default=1.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_output/nepal_hydrologic_foundation.png"),
    )
    return parser.parse_args()


def _central_crop(
    source: RasterModelGrid,
    rows: int,
    cols: int,
    row_offset: int = 0,
    col_offset: int = 0,
) -> RasterModelGrid:
    """Return a central metric-grid crop while preserving Landlab node order."""
    if rows < 3 or cols < 3:
        raise ValueError("The crop must contain at least three rows and columns")
    if rows > source.shape[0] or cols > source.shape[1]:
        raise ValueError(f"Crop {(rows, cols)} exceeds DEM shape {source.shape}")

    row0 = (source.shape[0] - rows) // 2 + row_offset
    col0 = (source.shape[1] - cols) // 2 + col_offset
    if row0 < 0 or col0 < 0 or row0 + rows > source.shape[0] or col0 + cols > source.shape[1]:
        raise ValueError("Requested crop offsets place the crop outside the DEM")

    source_z = source.at_node["topographic__elevation"].reshape(source.shape)
    cropped_z = source_z[row0 : row0 + rows, col0 : col0 + cols].copy()

    lower_left = (
        source.xy_of_lower_left[0] + col0 * source.dx,
        source.xy_of_lower_left[1] + row0 * source.dy,
    )
    grid = RasterModelGrid(
        (rows, cols),
        xy_spacing=(source.dx, source.dy),
        xy_of_lower_left=lower_left,
        xy_axis_units="m",
    )
    grid.add_field("topographic__elevation", cropped_z.ravel(), at="node")
    return grid


def _storm_wetness(
    grid: RasterModelGrid, background: float, peak: float
) -> np.ndarray:
    """Create a bounded synthetic storm footprint for exercising field mode."""
    if not 0.0 <= background <= peak <= 1.0:
        raise ValueError(
            "Wetness values must satisfy 0 <= background <= storm peak <= 1"
        )

    x = grid.node_x.reshape(grid.shape)
    y = grid.node_y.reshape(grid.shape)
    x_span = float(np.ptp(x))
    y_span = float(np.ptp(y))
    x_center = float(x.min() + 0.58 * x_span)
    y_center = float(y.min() + 0.55 * y_span)
    sigma_x = max(0.24 * x_span, grid.dx)
    sigma_y = max(0.30 * y_span, grid.dy)
    footprint = np.exp(
        -0.5
        * (
            ((x - x_center) / sigma_x) ** 2
            + ((y - y_center) / sigma_y) ** 2
        )
    )
    return (background + (peak - background) * footprint).ravel()


def _evaluate_stability(
    landslider: ShallowLandslider, wetness_field: np.ndarray, wetness: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the public pipeline and copy state that changes on the next call."""
    wetness_field[:] = wetness
    landslider.run_one_step()
    return (
        landslider.results["factor_of_safety"].copy(),
        landslider.results["unstable_mask"].copy(),
    )


def _plot_field(
    ax,
    values: np.ndarray,
    grid: RasterModelGrid,
    title: str,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "",
):
    data = np.asarray(values).reshape(grid.shape)
    extent = (
        0.0,
        (grid.shape[1] - 1) * grid.dx / 1000.0,
        0.0,
        (grid.shape[0] - 1) * grid.dy / 1000.0,
    )
    image = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Easting within crop (km)")
    ax.set_ylabel("Northing within crop (km)")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    return image


def main() -> None:
    args = _parse_args()
    if not args.dem.exists():
        raise FileNotFoundError(f"Nepal DEM not found: {args.dem}")

    full_grid, _, _ = get_topo(
        buffer=0.0,
        dem_type="SRTMGL1",
        load_dem=str(args.dem),
        grid_spacing=30.0,
    )
    grid = _central_crop(
        full_grid,
        args.rows,
        args.cols,
        row_offset=args.row_offset,
        col_offset=args.col_offset,
    )

    apply_soil_depth(
        grid,
        max_soil_depth=args.max_soil_depth,
        distribution="elevation",
        relationship="linear",
    )

    wetness_field = grid.add_full(
        "soil__relative_wetness", args.background_wetness, at="node"
    )
    landslider = ShallowLandslider(
        grid,
        cohesion_eff=args.cohesion,
        angle_int_frict=args.friction_angle,
        wetness_source="field",
        # PGA is intentionally omitted and therefore zero.
        random_seed=0,
    )

    background = np.full(grid.number_of_nodes, args.background_wetness)
    storm = _storm_wetness(
        grid, args.background_wetness, args.storm_peak_wetness
    )
    dry_fos, dry_unstable = _evaluate_stability(
        landslider, wetness_field, background
    )
    storm_fos, storm_unstable = _evaluate_stability(
        landslider, wetness_field, storm
    )
    critical_wetness = landslider.results["critical_relative_wetness"].copy()

    dry_count = int(np.count_nonzero(dry_unstable))
    storm_count = int(np.count_nonzero(storm_unstable))
    activated = storm_unstable & ~dry_unstable
    activated_count = int(np.count_nonzero(activated))
    core_count = int(grid.number_of_core_nodes)

    figure, axes = plt.subplots(2, 4, figsize=(19, 9), constrained_layout=True)
    _plot_field(
        axes[0, 0],
        grid.at_node["topographic__elevation"],
        grid,
        "Nepal DEM subregion",
        cmap="terrain",
        colorbar_label="Elevation (m)",
    )
    _plot_field(
        axes[0, 1],
        grid.at_node["soil__depth"],
        grid,
        "Modelled soil depth",
        cmap="YlOrBr",
        vmin=0.0,
        vmax=args.max_soil_depth,
        colorbar_label="Soil depth (m)",
    )
    _plot_field(
        axes[0, 2],
        critical_wetness,
        grid,
        "Critical relative wetness $m_c$",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="$m_c$ (display clipped to 0–1)",
    )
    _plot_field(
        axes[0, 3],
        storm,
        grid,
        "Prescribed synthetic storm wetness",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Relative wetness $m$",
    )
    _plot_field(
        axes[1, 0],
        dry_fos,
        grid,
        f"Background FoS ($m={args.background_wetness:.2f}$)",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=2.0,
        colorbar_label="Factor of safety",
    )
    _plot_field(
        axes[1, 1],
        storm_fos,
        grid,
        "Storm-state FoS",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=2.0,
        colorbar_label="Factor of safety",
    )
    _plot_field(
        axes[1, 2],
        dry_unstable.astype(float),
        grid,
        f"Background unstable ({dry_count:,} nodes)",
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Unstable mask",
    )
    _plot_field(
        axes[1, 3],
        activated.astype(float),
        grid,
        f"Storm-activated ({activated_count:,} nodes)",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Unstable in storm only",
    )

    figure.suptitle(
        "ShallowLandslider 2.0 hydrologic foundation — zero PGA\n"
        f"{grid.shape[0]}×{grid.shape[1]} nodes at {grid.dx:.0f} m; "
        f"cohesion={args.cohesion / 1000:.1f} kPa; "
        f"friction={args.friction_angle:.1f}°",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote {args.output}")
    print(f"Grid: {grid.shape[0]} x {grid.shape[1]} ({core_count:,} core nodes)")
    print(f"Background unstable: {dry_count:,}")
    print(f"Storm unstable: {storm_count:,}")
    print(f"Storm-activated: {activated_count:,}")
    print(
        "Hydrologically reachable core nodes (0 < m_c <= 1): "
        f"{np.count_nonzero((critical_wetness[grid.core_nodes] > 0.0) & (critical_wetness[grid.core_nodes] <= 1.0)):,}"
    )


if __name__ == "__main__":
    main()
