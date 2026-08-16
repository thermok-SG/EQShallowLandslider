"""Analysis helpers for self-describing ShallowLandslider v1.2 run bundles.

The public functions load individual or ensemble region results, normalize
measured inventory column aliases, create distribution and spatial-diagnostic
figures, summarize distributions, and calculate model/observation distances.
Raster loading is optional because ensemble table analysis should not incur the
cost of opening large Zarr or NumPy arrays.

Statistical comparisons operate on marginal distributions; they do not prove
spatial agreement or causal model validity. The Kuiper p-value is an asymptotic
two-sample approximation, while Wasserstein distance retains the displayed
variable's physical units.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from scipy.stats import ks_2samp, wasserstein_distance

_PLOT_COLUMNS = [
    ("area", "Area (m²)", True),
    ("median_slope", "Median slope (°)", False),
    ("median_elevation", "Median elevation (m)", False),
    ("local_relief", "Local relief (m)", False),
    ("slope_direction_length_new", "Length (m)", True),
    ("perpendicular_width_new", "Width (m)", True),
]

_OBSERVED_COLUMN_ALIASES = {
    "area": ("area_m2", "Area_m2", "Area", "area"),
    "median_slope": (
        "Slope_deg_mean",
        "Mean_slope",
        "mean_slope",
        "median_slope",
        "slope",
    ),
    "median_elevation": (
        "Elevation_m_mean",
        "Elevation_mean",
        "mean_elev",
        "median_elevation",
        "elevation",
    ),
    "slope_direction_length_new": ("length_m", "Length_m", "length"),
    "perpendicular_width_new": ("width_m", "Width_m", "width"),
}


def discover_runs(output_root):
    """Return run directories containing a v1.2 manifest, ordered by creation time."""
    root = Path(output_root)
    return sorted(path.parent for path in root.rglob("manifest.json"))


def _read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def load_run(run_dir, load_rasters=False, mmap_mode="r"):
    """Load one run's metadata and table, optionally opening its rasters lazily."""
    run_dir = Path(run_dir)
    manifest = _read_json(run_dir / "manifest.json")
    summary = _read_json(run_dir / "summary.json")
    parquet = run_dir / "regions.parquet"
    if parquet.exists():
        try:
            regions = pd.read_parquet(parquet)
        except (ImportError, ModuleNotFoundError):
            regions = pd.read_csv(run_dir / "regions.csv")
    else:
        regions = pd.read_csv(run_dir / "regions.csv")

    rasters = None
    if load_rasters:
        zarr_path = run_dir / "rasters.zarr"
        if zarr_path.exists():
            try:
                import xarray as xr

                rasters = xr.open_zarr(zarr_path, chunks=None)
            except (ImportError, ModuleNotFoundError) as exc:
                raise ImportError(
                    "xarray and zarr are required to load rasters.zarr"
                ) from exc
        else:
            raster_dir = run_dir / "rasters"
            metadata = _read_json(raster_dir / "metadata.json")
            rasters = {
                name: np.load(raster_dir / f"{name}.npy", mmap_mode=mmap_mode)
                for name in metadata["fields"]
            }
    return {
        "path": run_dir,
        "manifest": manifest,
        "summary": summary,
        "regions": regions,
        "rasters": rasters,
    }


def load_region_ensemble(output_root, selected_only=False):
    """Combine region tables from every discovered v1.2 run."""
    tables = []
    for run_dir in discover_runs(output_root):
        table = load_run(run_dir, load_rasters=False)["regions"]
        if selected_only and "selected" in table:
            table = table[table["selected"]]
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _parameter_label(path):
    """Return a compact plot label for a dotted configuration path."""
    labels = {
        "cohesion_eff": "Effective cohesion (Pa)",
        "angle_int_frict": "Internal friction angle (degrees)",
        "max_soil_depth": "Maximum soil depth (m)",
        "horizontal_max": "Maximum horizontal PGA (g)",
        "vertical_max": "Maximum vertical PGA (g)",
    }
    leaf = path.rsplit(".", 1)[-1]
    return labels.get(leaf, path)


def _parameter_name(path):
    """Return a short parameter name suitable for figure headings."""
    labels = {
        "cohesion_eff": "effective cohesion",
        "angle_int_frict": "internal friction angle",
        "submerged_soil_proportion": "submerged soil proportion",
        "max_soil_depth": "maximum soil depth",
        "horizontal_max": "maximum horizontal PGA",
        "vertical_max": "maximum vertical PGA",
    }
    leaf = path.rsplit(".", 1)[-1]
    return labels.get(leaf, leaf.replace("_", " "))


def _sentence_case(value):
    """Uppercase an initial character without lowercasing acronyms."""
    return value[:1].upper() + value[1:]


def _format_parameter_value(path, value):
    """Format a swept or fixed value compactly, including known units."""
    if isinstance(value, float):
        value_text = f"{value:g}"
    else:
        value_text = str(value)
    units = {
        "cohesion_eff": " Pa",
        "angle_int_frict": "°",
        "max_soil_depth": " m",
        "horizontal_max": " g",
        "vertical_max": " g",
    }
    return value_text + units.get(path.rsplit(".", 1)[-1], "")


def _safe_filename(value):
    safe = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(value)
    ).strip("-")
    return safe or "parameter"


def _json_key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _resolve_swept_parameter(runs, requested):
    available = sorted(
        {
            path
            for run in runs
            for path in run["manifest"]
            .get("config", {})
            .get("ensemble", {})
            .get("parameters", {})
        }
    )
    if requested in available:
        return requested
    matches = [path for path in available if path.rsplit(".", 1)[-1] == requested]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        choices = ", ".join(available) if available else "none"
        raise ValueError(
            f"{requested!r} is not a swept ensemble parameter. Available: {choices}"
        )
    raise ValueError(
        f"{requested!r} is ambiguous; use one of these dotted paths: "
        + ", ".join(matches)
    )


def _latest_ensemble_runs(output_root):
    """Load one completed run per ensemble digest, preferring the latest rerun."""
    latest = {}
    for run_dir in discover_runs(output_root):
        run = load_run(run_dir, load_rasters=False)
        ensemble = run["manifest"].get("config", {}).get("ensemble", {})
        if not isinstance(ensemble.get("parameters"), dict):
            continue
        digest = ensemble.get("config_digest")
        key = (ensemble.get("name", "ensemble"), digest or str(run_dir))
        created = run["manifest"].get("created_utc", "")
        if key not in latest or created > latest[key][0]:
            latest[key] = (created, run)
    return [item[1] for item in latest.values()]


def swept_parameters(output_root):
    """Return the sorted dotted parameter paths found in completed ensembles."""
    return sorted(
        {
            path
            for run in _latest_ensemble_runs(output_root)
            for path in run["manifest"]["config"]["ensemble"]["parameters"]
        }
    )


def plot_parameter_sensitivity(
    output_root,
    parameter,
    output_dir,
    selected_only=True,
    observed=None,
):
    """Create controlled ensemble comparisons for one swept parameter.

    Runs are grouped by ensemble name and by every other swept parameter, so
    each figure varies only ``parameter``. Within a panel, ECDFs show the full
    region distribution and legends retain the sample count, including zero;
    distribution summaries are written to the returned table. When measured
    data are supplied, a dashed reference ECDF is added to each compatible
    panel. One completed run per configuration digest is used; for forced
    reruns, the latest completed run is selected.

    Parameters
    ----------
    output_root : path-like
        Root containing completed ensemble run bundles.
    parameter : str
        Dotted swept path, or an unambiguous final component such as
        ``cohesion_eff``.
    output_dir : path-like
        Directory in which comparison PNGs are written.
    selected_only : bool, default True
        Compare selected landslides rather than all candidate regions.
    observed : pandas.DataFrame, optional
        Measured landslides normalized to the model region-column convention.

    Returns
    -------
    pandas.DataFrame
        One row per plotted run and metric, including the varied value, fixed
        controls, sample count, median, and figure path.
    """
    runs = _latest_ensemble_runs(output_root)
    if not runs:
        raise ValueError(f"No completed ensemble runs found beneath {output_root}")
    parameter = _resolve_swept_parameter(runs, parameter)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = {}
    for run in runs:
        ensemble = run["manifest"]["config"]["ensemble"]
        parameters = ensemble["parameters"]
        if parameter not in parameters:
            continue
        controls = {key: value for key, value in parameters.items() if key != parameter}
        key = (ensemble.get("name", "ensemble"), _json_key(controls))
        groups.setdefault(key, {"controls": controls, "runs": []})["runs"].append(
            (parameters[parameter], run)
        )

    records = []
    figure_number = 0
    for (ensemble_name, _), group in sorted(groups.items(), key=lambda item: item[0]):
        varied = group["runs"]
        distinct_values = {_json_key(value) for value, _run in varied}
        if len(distinct_values) < 2:
            continue
        try:
            varied.sort(key=lambda item: (float(item[0]), str(item[0])))
        except (TypeError, ValueError):
            varied.sort(key=lambda item: str(item[0]))

        figure_number += 1
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
        colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(varied)))
        figure_path = output_dir / (
            f"{_safe_filename(parameter)}_{figure_number:03d}.png"
        )
        for axis, (column, metric_label, log_x) in zip(axes.ravel(), _PLOT_COLUMNS):
            for color, (value, run) in zip(colors, varied):
                regions = run["regions"]
                if selected_only and "selected" in regions:
                    regions = regions[regions["selected"]]
                values = _numeric_values(regions, column, positive=log_x)
                value_label = (
                    f"{value:g}" if isinstance(value, (int, float)) else str(value)
                )
                line_label = f"{value_label} (n={len(values):,})"
                if len(values):
                    ecdf_x, ecdf_y = _ecdf(values)
                else:
                    ecdf_x = ecdf_y = np.array([])
                axis.plot(
                    ecdf_x,
                    ecdf_y,
                    color=color,
                    linewidth=1.8,
                    label=line_label,
                )
                records.append(
                    {
                        "ensemble": ensemble_name,
                        "comparison_id": figure_number,
                        "parameter": parameter,
                        "parameter_value": value,
                        "fixed_parameters": _json_key(group["controls"]),
                        "run_id": run["summary"]["run_id"],
                        "metric": column,
                        "count": len(values),
                        "median": float(np.median(values)) if len(values) else np.nan,
                        "mean": float(np.mean(values)) if len(values) else np.nan,
                        "p05": (
                            float(np.quantile(values, 0.05)) if len(values) else np.nan
                        ),
                        "p95": (
                            float(np.quantile(values, 0.95)) if len(values) else np.nan
                        ),
                        "figure": str(figure_path),
                    }
                )
            observed_values = (
                _numeric_values(observed, column, positive=log_x)
                if observed is not None
                else np.array([])
            )
            if len(observed_values):
                observed_x, observed_y = _ecdf(observed_values)
                axis.plot(
                    observed_x,
                    observed_y,
                    color="black",
                    linestyle="--",
                    linewidth=2.2,
                    label=f"Measured (n={len(observed_values):,})",
                    zorder=10,
                )
            if log_x:
                axis.set_xscale("log")
            axis.set_xlabel(metric_label)
            axis.set_ylabel("ECDF")
            axis.set_ylim(0, 1)
            axis.grid(alpha=0.2)
            axis.set_title(metric_label.split(" (")[0], fontsize=11)

        legend_entries = {}
        for axis in axes.ravel():
            handles, labels = axis.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_entries.setdefault(label, handle)
        labels = list(legend_entries)
        handles = [legend_entries[label] for label in labels]
        if handles:
            fig.legend(
                handles,
                labels,
                title=_parameter_label(parameter),
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
            )
        controls_text = (
            " · ".join(
                f"{_sentence_case(_parameter_name(key))} = "
                f"{_format_parameter_value(key, value)}"
                for key, value in group["controls"].items()
            )
            or "Other swept parameters fixed"
        )
        selection = "Selected" if selected_only else "Candidate"
        ensemble_label = ensemble_name.replace("_", " ")
        fixed_lines = "\n".join(
            textwrap.wrap(
                f"Fixed: {controls_text}",
                width=92,
                subsequent_indent="       ",
            )
        )
        fig.suptitle(
            f"Sensitivity to {_sentence_case(_parameter_name(parameter))}\n"
            f"{selection} landslides · {ensemble_label}\n"
            f"{fixed_lines}",
            fontsize=14,
            fontweight="semibold",
            linespacing=1.35,
        )
        fig.savefig(figure_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    if not figure_number:
        raise ValueError(
            f"No controlled comparison for {parameter!r} has at least two values"
        )
    return pd.DataFrame.from_records(records)


def _first_column(table, aliases):
    return next((column for column in aliases if column in table.columns), None)


def _filter_observed_area(table, min_area):
    if min_area is None:
        return table
    area_column = _first_column(table, _OBSERVED_COLUMN_ALIASES["area"])
    if area_column is None:
        return table
    area = pd.to_numeric(table[area_column], errors="coerce")
    return table[area >= min_area]


def load_observed_landslides(inventory_path=None, zonal_stats_path=None, min_area=None):
    """Load measured inventory CSVs into the model region-column convention.

    The inventory supplies area, length, and width; the zonal-statistics file
    supplies area, slope, and elevation. Either file may be omitted. Columns
    unavailable in the supplied files are left absent and remain model-only in
    comparison plots.

    Parameters
    ----------
    inventory_path : path-like, optional
        CSV containing measured area/length/width records.
    zonal_stats_path : path-like, optional
        CSV containing measured terrain summaries. Common project aliases are
        normalized automatically.
    min_area : float, optional
        Minimum observed area in square metres. Filtering is applied separately
        to each supplied file using its own area column.

    Returns
    -------
    pandas.DataFrame
        Available observed variables renamed to the model region schema. Files
        with different row counts remain independent marginal samples; rows are
        not interpreted as cross-file joins.
    """
    if inventory_path is None and zonal_stats_path is None:
        raise ValueError("Provide an inventory or zonal-statistics CSV.")

    inventory = None
    if inventory_path is not None:
        inventory = pd.read_csv(inventory_path, encoding="utf-8-sig")
        inventory = _filter_observed_area(inventory, min_area).reset_index(drop=True)

    zonal_stats = None
    if zonal_stats_path is not None:
        zonal_stats = pd.read_csv(zonal_stats_path, encoding="utf-8-sig")
        zonal_stats = _filter_observed_area(zonal_stats, min_area).reset_index(
            drop=True
        )

    columns = {}
    sources = {
        "area": (inventory, zonal_stats),
        "median_slope": (zonal_stats, inventory),
        "median_elevation": (zonal_stats, inventory),
        "slope_direction_length_new": (inventory, zonal_stats),
        "perpendicular_width_new": (inventory, zonal_stats),
    }
    for target, tables in sources.items():
        for table in tables:
            if table is None:
                continue
            source = _first_column(table, _OBSERVED_COLUMN_ALIASES[target])
            if source is not None:
                columns[target] = pd.to_numeric(table[source], errors="coerce")
                break
    return pd.DataFrame(columns)


def _ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    return values, (
        np.arange(1, len(values) + 1) / len(values) if len(values) else values
    )


def _shared_bins(values, log_x):
    """Return valid shared histogram edges, including for constant samples."""
    minimum = values.min()
    maximum = values.max()
    if minimum == maximum:
        if log_x:
            return np.geomspace(minimum / 1.1, maximum * 1.1, 21)
        padding = max(abs(minimum) * 0.05, 0.5)
        return np.linspace(minimum - padding, maximum + padding, 21)
    if log_x:
        return np.geomspace(minimum, maximum, 21)
    return np.histogram_bin_edges(values, bins="auto")


def _numeric_values(table, column, positive=False):
    values = pd.to_numeric(table.get(column, pd.Series(dtype=float)), errors="coerce")
    values = values.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values[values > 0] if positive else values


def summarize_run_distributions(run, observed=None, selected_only=True):
    """Return descriptive statistics for model and optional measured data.

    Statistics include count, mean, median, population standard deviation, and
    5th/95th percentiles for every available plotted metric. When
    ``selected_only`` is true, candidate regions not selected by the model are
    excluded.
    """
    if not isinstance(run, dict):
        run = load_run(run)
    regions = run["regions"]
    if selected_only and "selected" in regions:
        regions = regions[regions["selected"]]

    records = []
    for column, label, positive in _PLOT_COLUMNS:
        for source, table in (("model", regions), ("observed", observed)):
            if table is None or column not in table:
                continue
            values = _numeric_values(table, column, positive=positive)
            if not len(values):
                continue
            records.append(
                {
                    "run_id": run["summary"]["run_id"],
                    "source": source,
                    "metric": column,
                    "units_label": label,
                    "count": len(values),
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "p05": np.quantile(values, 0.05),
                    "p95": np.quantile(values, 0.95),
                }
            )
    return pd.DataFrame.from_records(records)


def _kuiper_2samp(first, second):
    """Return the two-sample Kuiper statistic and its asymptotic p-value."""
    first = np.sort(first)
    second = np.sort(second)
    points = np.sort(np.concatenate((first, second)))
    difference = np.searchsorted(first, points, side="right") / len(
        first
    ) - np.searchsorted(second, points, side="right") / len(second)
    statistic = float(difference.max() - difference.min())
    if statistic == 0:
        return statistic, 1.0

    effective_n = len(first) * len(second) / (len(first) + len(second))
    root_n = np.sqrt(effective_n)
    scaled = (root_n + 0.155 + 0.24 / root_n) * statistic
    terms = np.arange(1, 101, dtype=float)
    p_value = 2 * np.sum(
        (4 * terms**2 * scaled**2 - 1) * np.exp(-2 * terms**2 * scaled**2)
    )
    return statistic, float(np.clip(p_value, 0, 1))


def compare_run_distributions(run, observed, selected_only=True):
    """Compare model and measured marginal distributions.

    Returns one row per shared metric with sample counts, the two-sample
    Kolmogorov–Smirnov statistic/p-value, the two-sample Kuiper
    statistic/asymptotic p-value, and first Wasserstein distance. Wasserstein
    values retain the metric's units. These are distributional rather than
    spatial tests and do not establish model validity by themselves.
    """
    if not isinstance(run, dict):
        run = load_run(run)
    regions = run["regions"]
    if selected_only and "selected" in regions:
        regions = regions[regions["selected"]]

    records = []
    for column, label, positive in _PLOT_COLUMNS:
        if column not in regions or column not in observed:
            continue
        model_values = _numeric_values(regions, column, positive=positive)
        observed_values = _numeric_values(observed, column, positive=positive)
        if not len(model_values) or not len(observed_values):
            continue
        ks_result = ks_2samp(model_values, observed_values, method="auto")
        kuiper_statistic, kuiper_pvalue = _kuiper_2samp(model_values, observed_values)
        records.append(
            {
                "run_id": run["summary"]["run_id"],
                "metric": column,
                "units_label": label,
                "model_count": len(model_values),
                "observed_count": len(observed_values),
                "ks_statistic": ks_result.statistic,
                "ks_pvalue": ks_result.pvalue,
                "kuiper_statistic": kuiper_statistic,
                "kuiper_pvalue_approx": kuiper_pvalue,
                "wasserstein_distance": wasserstein_distance(
                    model_values, observed_values
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _raster_array(rasters, name):
    if rasters is None:
        return None
    if isinstance(rasters, dict):
        value = rasters.get(name)
    else:
        value = rasters[name] if name in rasters else None
    return None if value is None else np.asarray(value)


def _finite_limits(values, percentiles=(2, 98)):
    values = np.ma.asarray(values).compressed()
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None
    lower, upper = np.percentile(finite, percentiles)
    if lower == upper:
        padding = max(abs(lower) * 0.05, 1e-12)
        return lower - padding, upper + padding
    return lower, upper


def plot_run_maps(run, output_path=None, show=False):
    """Plot available spatial inputs, stability results, and runout fields.

    The adaptive three-column figure includes elevation, derived terrain slope,
    soil depth, PGA, factor of safety, unstable/selected footprints, Newmark
    displacement, and any saved erosion/deposition fields. Footprints are
    draped on grayscale elevation. Missing optional fields are omitted.

    ``run`` may be a run path or a loaded run dictionary. Rasters are loaded
    automatically if needed. The created Matplotlib figure is returned for
    further customization; callers processing ensembles should close it after
    saving.
    """
    if not isinstance(run, dict) or run.get("rasters") is None:
        run = load_run(run["path"] if isinstance(run, dict) else run, load_rasters=True)
    rasters = run["rasters"]
    elevation = _raster_array(rasters, "topographic_elevation")
    if elevation is None:
        raise ValueError("Map plots require the topographic_elevation raster.")

    dx = float(run["manifest"]["grid"]["dx"])
    dy = float(run["manifest"]["grid"]["dy"])
    elevation_dy, elevation_dx = np.gradient(elevation.astype(float), dy, dx)
    terrain_slope = np.degrees(np.arctan(np.hypot(elevation_dx, elevation_dy)))
    extent = (
        0,
        (elevation.shape[1] - 1) * dx / 1000,
        0,
        (elevation.shape[0] - 1) * dy / 1000,
    )
    panels = [
        ("topographic_elevation", "Elevation", "terrain", "m", "continuous"),
        ("terrain_slope", "Terrain slope", "magma", "degrees", "continuous"),
        ("soil_depth", "Soil depth", "YlGnBu", "m", "continuous"),
        ("horizontal_pga", "Horizontal PGA", "magma", "g", "continuous"),
        ("factor_of_safety", "Factor of safety", "RdYlGn", "", "factor_safety"),
        ("unstable_mask", "Unstable cells", "Oranges", "", "footprint"),
        ("selected_footprint", "Selected landslides", "Reds", "", "footprint"),
        (
            "newmark_displacement",
            "Newmark displacement",
            "viridis",
            "m",
            "positive_log",
        ),
        ("runout_erosion", "Runout erosion", "Reds", "m", "positive"),
        ("runout_deposition", "Runout deposition", "Blues", "m", "positive"),
        (
            "runout_soil_depth_change",
            "Net soil-depth change",
            "RdBu",
            "m",
            "diverging",
        ),
        (
            "combined_affected_footprint",
            "Combined affected footprint",
            "Purples",
            "",
            "footprint",
        ),
    ]
    available = [
        panel
        for panel in panels
        if panel[0] == "terrain_slope" or _raster_array(rasters, panel[0]) is not None
    ]
    ncols = 3
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), layout="constrained"
    )
    axes = np.atleast_1d(axes).ravel()

    for axis, (name, title, cmap, units, style) in zip(axes, available):
        values = (
            terrain_slope
            if name == "terrain_slope"
            else _raster_array(rasters, name).astype(float)
        )
        if style == "footprint":
            axis.imshow(elevation, origin="lower", extent=extent, cmap="gray")
            mask = np.ma.masked_where(~values.astype(bool), values)
            axis.imshow(
                mask,
                origin="lower",
                extent=extent,
                cmap=cmap,
                alpha=0.75,
                vmin=0,
                vmax=1,
            )
            count = int(np.count_nonzero(values))
            axis.set_title(f"{title} (n={count:,} cells)")
        else:
            if style in {"positive", "positive_log"}:
                values = np.ma.masked_where(values <= 0, values)
            vmin, vmax = _finite_limits(values)
            norm = None
            if style == "positive_log" and vmin is not None and vmin > 0:
                norm = LogNorm(vmin=vmin, vmax=vmax)
                vmin = vmax = None
            elif style == "factor_safety" and vmin is not None:
                vmin = min(vmin, 0.99)
                vmax = 2.5
                norm = TwoSlopeNorm(vmin=vmin, vcenter=1, vmax=vmax)
                vmin = vmax = None
            elif style == "diverging" and vmin is not None:
                limit = max(abs(vmin), abs(vmax))
                norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
                vmin = vmax = None
            image = axis.imshow(
                values,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
            )
            colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
            if units:
                colorbar.set_label(units)
            axis.set_title(title)
        axis.set_xlabel("Easting distance (km)")
        axis.set_ylabel("Northing distance (km)")
        axis.set_aspect("equal")

    for axis in axes[len(available) :]:
        axis.set_visible(False)
    fig.suptitle(f"{run['summary']['run_id']} — spatial diagnostics")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    return fig


def plot_run(run, selected_only=True, observed=None, output_path=None, show=False):
    """Plot modeled distributions with optional measured-landslide overlays.

    Histograms use shared bins and each panel includes a same-color ECDF. Area,
    length, and width use logarithmic x axes. Sample counts are shown in the
    legend, and metrics absent from the observed table remain model-only.
    """
    if not isinstance(run, dict):
        run = load_run(run)
    regions = run["regions"]
    if selected_only and "selected" in regions:
        regions = regions[regions["selected"]]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for axis, (column, label, log_x) in zip(axes.ravel(), _PLOT_COLUMNS):
        model_values = _numeric_values(regions, column, positive=log_x)
        observed_values = (
            _numeric_values(observed, column, positive=log_x)
            if observed is not None
            else np.array([])
        )
        combined = np.concatenate((model_values, observed_values))
        if len(combined):
            bins = _shared_bins(combined, log_x)
            if len(model_values):
                axis.hist(
                    model_values,
                    bins=bins,
                    alpha=0.5,
                    color="tab:blue",
                    density=True,
                    label=f"Model (n={len(model_values):,})",
                )
            if len(observed_values):
                axis.hist(
                    observed_values,
                    bins=bins,
                    alpha=0.45,
                    color="tab:orange",
                    density=True,
                    label=f"Measured (n={len(observed_values):,})",
                )
            ecdf_axis = axis.twinx()
            for values, color, linestyle in (
                (model_values, "tab:blue", "-"),
                (observed_values, "tab:orange", "--"),
            ):
                if len(values):
                    ecdf_x, ecdf_y = _ecdf(values)
                    ecdf_axis.plot(
                        ecdf_x, ecdf_y, color=color, linestyle=linestyle, linewidth=1.5
                    )
            ecdf_axis.set_ylim(0, 1)
            ecdf_axis.set_ylabel("ECDF")
            axis.legend(fontsize="small")
        if log_x:
            axis.set_xscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
    selection = "selected" if selected_only else "candidate"
    fig.suptitle(f"{run['summary']['run_id']} — {selection} regions")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
    if show:
        plt.show()
    return fig
