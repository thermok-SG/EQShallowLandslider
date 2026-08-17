#!/usr/bin/env python3
"""Generate a reproducible, analysis-ready synthetic mountain range.

The generator evolves bedrock and sediment on a coarse Landlab grid using D8
priority-flood routing and ``SpaceLargeScaleEroder``, then cubic-resamples the
result to the requested model resolution. It writes matched ESRI ASCII
topographic-elevation, soil-depth, and bedrock-elevation grids plus a JSON
provenance/diagnostics file. The default command is::

    python generate_synthetic_topography.py

Use this surface for controlled model experiments, not as a statistical
surrogate for a particular real landscape. Changing the seed changes the
initial microtopography; changing geomorphic parameters changes the landscape
evolution experiment itself. Existing output files are replaced when the same
``--output`` path is reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from landlab import RasterModelGrid
from landlab.components import (
    DepthDependentTaylorDiffuser,
    ExponentialWeatherer,
    PriorityFloodFlowRouter,
    SpaceLargeScaleEroder,
)
from landlab.io import esri_ascii
from scipy.ndimage import zoom


class _RawDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Preserve examples while displaying argument defaults."""


def _mountain_belt_uplift(grid):
    """Return normalized uplift highest along an elongated axial divide."""
    x_fraction = grid.node_x / grid.node_x.max()
    y_fraction = grid.node_y / grid.node_y.max()
    uplift = np.sin(np.pi * x_fraction) * (
        0.85 + 0.15 * np.sin(np.pi * y_fraction)
    )
    return uplift / uplift.max()


def _resample_to_fine_grid(source_grid, nrows, ncols, spacing, factor):
    """Cubic-resample evolved fields onto the requested 30 m Landlab grid."""
    fine_grid = RasterModelGrid(
        (nrows, ncols), xy_spacing=spacing, xy_axis_units="m"
    )
    for name in ("bedrock__elevation", "soil__depth", "topographic__elevation"):
        source = source_grid.at_node[name].reshape(source_grid.shape)
        values = zoom(source, factor, order=3, mode="nearest", prefilter=True)
        if values.shape != fine_grid.shape:
            raise RuntimeError(
                f"Resampled {name} has shape {values.shape}, expected {fine_grid.shape}"
            )
        if name == "soil__depth":
            values = np.maximum(values, 0.0)
        fine_grid.add_field(name, values.ravel(), at="node", copy=True)

    # Enforce exact consistency after interpolation.
    fine_grid.at_node["topographic__elevation"][:] = (
        fine_grid.at_node["bedrock__elevation"] + fine_grid.at_node["soil__depth"]
    )
    return fine_grid


def generate_mountain_catchment(
    nrows=600,
    ncols=800,
    spacing=30.0,
    seed=20260804,
    outlet_elevation=450.0,
    iterations=1000,
    timestep=250.0,
    uplift_rate=0.001,
    rock_erodibility=2.5e-5,
    sediment_erodibility=2.5e-5,
    initial_soil_depth=0.5,
    refinement_factor=2,
    regolith_model="space",
    soil_production_maximum_rate=1.0e-4,
    soil_production_decay_depth=0.5,
    soil_transport_velocity=0.02,
    soil_transport_decay_depth=0.5,
    critical_slope=1.0,
):
    """Evolve and return a tutorial-based synthetic mountain range.

    The method follows Landlab's PriorityFlood landscape-evolution and HyLands
    tutorials: low-relief random initial topography, spatially variable uplift,
    priority-flood D8 routing, and SPACE bedrock/alluvial evolution. All outer
    boundaries remain fixed at base level, allowing multiple drainage basins to
    form naturally on both sides of an elongated axial uplift zone.

    Evolution occurs on a coarser grid and is cubically resampled to the final
    grid. This preserves the requested regional extent and 30 m resolution while
    making a long geomorphic spin-up practical for ensemble preparation.

    ``regolith_model="space"`` preserves SPACE's mobile-sediment layer.
    ``regolith_model="weathering_taylor"`` additionally couples exponential
    bedrock weathering to depth-dependent nonlinear Taylor soil creep, so the
    returned surface, bedrock, and regolith fields co-evolve.
    """
    if nrows < 20 or ncols < 20:
        raise ValueError("nrows and ncols must each be at least 20")
    if spacing <= 0 or timestep <= 0:
        raise ValueError("spacing and timestep must be positive")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not isinstance(refinement_factor, int) or refinement_factor < 1:
        raise ValueError("refinement_factor must be a positive integer")
    if nrows % refinement_factor or ncols % refinement_factor:
        raise ValueError("nrows and ncols must be divisible by refinement_factor")
    if nrows // refinement_factor < 20 or ncols // refinement_factor < 20:
        raise ValueError("the evolution grid must have at least 20 rows and columns")
    if uplift_rate < 0 or min(rock_erodibility, sediment_erodibility) <= 0:
        raise ValueError(
            "uplift_rate cannot be negative; erodibilities must be positive"
        )
    if initial_soil_depth < 0:
        raise ValueError("initial_soil_depth cannot be negative")
    if regolith_model not in {"space", "weathering_taylor"}:
        raise ValueError("regolith_model must be 'space' or 'weathering_taylor'")
    if min(soil_production_maximum_rate, soil_production_decay_depth) <= 0:
        raise ValueError("soil-production parameters must be positive")
    if min(soil_transport_velocity, soil_transport_decay_depth, critical_slope) <= 0:
        raise ValueError(
            "soil-transport parameters and critical_slope must be positive"
        )

    source_shape = (nrows // refinement_factor, ncols // refinement_factor)
    source_spacing = spacing * refinement_factor
    source_grid = RasterModelGrid(
        source_shape, xy_spacing=source_spacing, xy_axis_units="m"
    )
    rng = np.random.default_rng(seed)

    # Tutorial-style microtopography initiates drainage convergence without
    # prescribing valleys, catchments, or the final drainage divide.
    bedrock = source_grid.add_field(
        "bedrock__elevation",
        rng.uniform(0.0, 0.1, source_grid.number_of_nodes),
        at="node",
        copy=True,
    )
    soil = source_grid.add_zeros("soil__depth", at="node")
    soil[source_grid.core_nodes] = initial_soil_depth
    elevation = source_grid.add_field(
        "topographic__elevation", bedrock + soil, at="node", copy=True
    )
    uplift_pattern = _mountain_belt_uplift(source_grid)

    flow = PriorityFloodFlowRouter(
        source_grid,
        flow_metric="D8",
        update_flow_depressions=True,
        accumulate_flow=True,
        suppress_out=True,
    )
    flow.run_one_step()
    space = SpaceLargeScaleEroder(
        source_grid,
        K_sed=sediment_erodibility,
        K_br=rock_erodibility,
        F_f=0.0,
        phi=0.0,
        H_star=1.0,
        v_s=1.0,
        v_s_lake=1.0,
        m_sp=0.5,
        n_sp=1.0,
        sp_crit_sed=0.0,
        sp_crit_br=0.0,
    )
    weatherer = None
    hillslope = None
    if regolith_model == "weathering_taylor":
        weatherer = ExponentialWeatherer(
            source_grid,
            soil_production_maximum_rate=soil_production_maximum_rate,
            soil_production_decay_depth=soil_production_decay_depth,
        )
        hillslope = DepthDependentTaylorDiffuser(
            source_grid,
            soil_transport_velocity=soil_transport_velocity,
            soil_transport_decay_depth=soil_transport_decay_depth,
            slope_crit=critical_slope,
            dynamic_dt=True,
            if_unstable="raise",
        )

    for _ in range(iterations):
        bedrock[source_grid.core_nodes] += (
            uplift_pattern[source_grid.core_nodes] * uplift_rate * timestep
        )
        elevation[:] = bedrock + soil
        flow.run_one_step()
        space.run_one_step(timestep)
        if weatherer is not None:
            # SPACE handles fluvial sediment; weathering and nonlinear creep
            # add an explicit hillslope regolith-production/transport process.
            weatherer.run_one_step()
            hillslope.run_one_step(timestep)

    fine_grid = _resample_to_fine_grid(
        source_grid, nrows, ncols, spacing, refinement_factor
    )
    elevation = fine_grid.at_node["topographic__elevation"]
    bedrock = fine_grid.at_node["bedrock__elevation"]
    soil = fine_grid.at_node["soil__depth"]

    # Keep the lowest open-boundary elevation at the requested datum.
    elevation += outlet_elevation - float(elevation.min())
    bedrock[:] = elevation - soil
    if (
        not np.isfinite(elevation).all()
        or not np.isfinite(soil).all()
        or float(np.ptp(elevation)) > 10_000.0
        or float(soil.max()) > 100.0
        or float(soil.min()) < -1.0e-8
    ):
        raise RuntimeError(
            "SPACE produced a nonphysical surface; reduce --timestep or adjust "
            "the evolution parameters"
        )

    # One fine-grid routing pass provides diagnostics at the actual model scale.
    fine_flow = PriorityFloodFlowRouter(
        fine_grid,
        flow_metric="D8",
        update_flow_depressions=True,
        accumulate_flow=True,
        suppress_out=True,
    )
    fine_flow.run_one_step()
    slopes = np.degrees(fine_grid.calc_slope_at_node(elevation, method="Horn"))
    valid_slopes = slopes[fine_grid.core_nodes]
    drainage_area = fine_grid.at_node["drainage_area"]

    component_names = [
        "PriorityFloodFlowRouter (D8)",
        "SpaceLargeScaleEroder",
    ]
    if regolith_model == "weathering_taylor":
        component_names.extend(["ExponentialWeatherer", "DepthDependentTaylorDiffuser"])
    stats = {
        "generator": "Landlab tutorial-based mountain-range evolution",
        "components": component_names,
        "regolith_model": regolith_model,
        "boundary_condition": "fixed-value open perimeter",
        "shape": [int(nrows), int(ncols)],
        "source_evolution_shape": [int(source_shape[0]), int(source_shape[1])],
        "refinement_factor": int(refinement_factor),
        "cell_count": int(nrows * ncols),
        "valid_cell_count": int(nrows * ncols),
        "spacing_m": float(spacing),
        "source_spacing_m": float(source_spacing),
        "width_m": float((ncols - 1) * spacing),
        "height_m": float((nrows - 1) * spacing),
        "seed": int(seed),
        "evolution": {
            "iterations": int(iterations),
            "timestep_years": float(timestep),
            "duration_years": float(iterations * timestep),
            "uplift_rate_m_per_year": float(uplift_rate),
            "rock_erodibility": float(rock_erodibility),
            "sediment_erodibility": float(sediment_erodibility),
            "initial_soil_depth_m": float(initial_soil_depth),
            "uplift_pattern": "elongated axial sine belt",
        },
        "elevation_min_m": float(elevation.min()),
        "elevation_max_m": float(elevation.max()),
        "elevation_relief_m": float(np.ptp(elevation)),
        "slope_degrees": {
            "p05": float(np.percentile(valid_slopes, 5)),
            "p50": float(np.percentile(valid_slopes, 50)),
            "p95": float(np.percentile(valid_slopes, 95)),
            "maximum": float(valid_slopes.max()),
        },
        "maximum_drainage_area_m2": float(drainage_area.max()),
        "final_soil_depth_m": {
            "p50": float(np.percentile(soil[fine_grid.core_nodes], 50)),
            "p95": float(np.percentile(soil[fine_grid.core_nodes], 95)),
            "maximum": float(soil[fine_grid.core_nodes].max()),
        },
    }
    if regolith_model == "weathering_taylor":
        stats["regolith_parameters"] = {
            "soil_production_maximum_rate_m_per_year": float(
                soil_production_maximum_rate
            ),
            "soil_production_decay_depth_m": float(soil_production_decay_depth),
            "soil_transport_velocity_m_per_year": float(soil_transport_velocity),
            "soil_transport_decay_depth_m": float(soil_transport_decay_depth),
            "critical_slope_m_per_m": float(critical_slope),
        }
    return fine_grid, stats


def write_esri_ascii(
    path, grid, field_name="topographic__elevation", nodata_value=-9999.0
):
    """Export a node field from the final grid to ESRI ASCII."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        esri_ascii.dump(
            grid,
            stream=stream,
            at="node",
            name=field_name,
            nodata_value=nodata_value,
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_RawDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="input_data/dem/synthetic_landlab_600x800_30m.asc",
        help="Output ESRI ASCII path.",
    )
    parser.add_argument("--nrows", type=int, default=600, help="Final grid rows.")
    parser.add_argument("--ncols", type=int, default=800, help="Final grid columns.")
    parser.add_argument(
        "--spacing", type=float, default=30.0, help="Final node spacing in metres."
    )
    parser.add_argument(
        "--seed", type=int, default=20260804, help="Initial roughness random seed."
    )
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Landscape-evolution steps."
    )
    parser.add_argument(
        "--timestep", type=float, default=250.0, help="Years per evolution step."
    )
    parser.add_argument(
        "--uplift-rate", type=float, default=0.001, help="Maximum uplift rate (m/yr)."
    )
    parser.add_argument(
        "--rock-erodibility",
        type=float,
        default=2.5e-5,
        help="SPACE bedrock erodibility coefficient.",
    )
    parser.add_argument(
        "--sediment-erodibility",
        type=float,
        default=2.5e-5,
        help="SPACE sediment erodibility coefficient.",
    )
    parser.add_argument(
        "--initial-soil-depth",
        type=float,
        default=0.5,
        help="Initial coarse-grid soil depth in metres.",
    )
    parser.add_argument(
        "--refinement-factor",
        type=int,
        default=2,
        help="Ratio of final to evolution-grid resolution; dimensions must divide evenly.",
    )
    parser.add_argument(
        "--regolith-model",
        choices=("space", "weathering_taylor"),
        default="space",
        help="SPACE-only sediment or SPACE plus weathering and nonlinear soil creep.",
    )
    parser.add_argument(
        "--soil-production-maximum-rate",
        type=float,
        default=1.0e-4,
        help="Bare-bedrock soil production rate (m/yr) for weathering_taylor.",
    )
    parser.add_argument(
        "--soil-production-decay-depth",
        type=float,
        default=0.5,
        help="E-folding soil-production depth (m) for weathering_taylor.",
    )
    parser.add_argument(
        "--soil-transport-velocity",
        type=float,
        default=0.02,
        help="Taylor soil-transport velocity (m/yr) for weathering_taylor.",
    )
    parser.add_argument(
        "--soil-transport-decay-depth",
        type=float,
        default=0.5,
        help="Taylor soil-transport depth scale (m) for weathering_taylor.",
    )
    parser.add_argument(
        "--critical-slope",
        type=float,
        default=1.0,
        help="Critical gradient (m/m) for nonlinear Taylor soil transport.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    grid, stats = generate_mountain_catchment(
        nrows=args.nrows,
        ncols=args.ncols,
        spacing=args.spacing,
        seed=args.seed,
        iterations=args.iterations,
        timestep=args.timestep,
        uplift_rate=args.uplift_rate,
        rock_erodibility=args.rock_erodibility,
        sediment_erodibility=args.sediment_erodibility,
        initial_soil_depth=args.initial_soil_depth,
        refinement_factor=args.refinement_factor,
        regolith_model=args.regolith_model,
        soil_production_maximum_rate=args.soil_production_maximum_rate,
        soil_production_decay_depth=args.soil_production_decay_depth,
        soil_transport_velocity=args.soil_transport_velocity,
        soil_transport_decay_depth=args.soil_transport_decay_depth,
        critical_slope=args.critical_slope,
    )
    output_path = Path(args.output)
    write_esri_ascii(output_path, grid)
    soil_path = output_path.with_name(f"{output_path.stem}_soil_depth.asc")
    bedrock_path = output_path.with_name(f"{output_path.stem}_bedrock_elevation.asc")
    write_esri_ascii(soil_path, grid, field_name="soil__depth")
    write_esri_ascii(bedrock_path, grid, field_name="bedrock__elevation")
    stats["outputs"] = {
        "topographic_elevation": str(output_path),
        "soil_depth": str(soil_path),
        "bedrock_elevation": str(bedrock_path),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} ({stats['shape'][0]} x {stats['shape'][1]} cells)")
    print(f"Wrote {soil_path}")
    print(f"Wrote {bedrock_path}")
    print(f"Wrote {metadata_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
