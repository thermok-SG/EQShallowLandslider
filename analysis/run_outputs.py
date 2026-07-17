"""Load, combine, and plot ShallowLandslider v1.2 run directories."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def discover_runs(output_root):
    """Return run directories containing a v1.2 manifest, ordered by creation time."""
    root = Path(output_root)
    return sorted(path.parent for path in root.glob("*/manifest.json"))


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
                raise ImportError("xarray and zarr are required to load rasters.zarr") from exc
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


def _ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    return values, np.arange(1, len(values) + 1) / len(values) if len(values) else values


def plot_run(run, selected_only=True, output_path=None, show=False):
    """Plot the principal distributions for one loaded run."""
    if not isinstance(run, dict):
        run = load_run(run)
    regions = run["regions"]
    if selected_only and "selected" in regions:
        regions = regions[regions["selected"]]

    columns = [
        ("area", "Area (m²)", True),
        ("median_slope", "Median slope (°)", False),
        ("median_elevation", "Median elevation (m)", False),
        ("local_relief", "Local relief (m)", False),
        ("slope_direction_length_new", "Length (m)", True),
        ("perpendicular_width_new", "Width (m)", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for axis, (column, label, log_x) in zip(axes.ravel(), columns):
        values = pd.to_numeric(regions.get(column, pd.Series(dtype=float)), errors="coerce")
        values = values[np.isfinite(values)]
        if log_x:
            values = values[values > 0]
        if len(values):
            axis.hist(values, bins="auto", alpha=0.65, color="steelblue", density=True)
            ecdf_x, ecdf_y = _ecdf(values)
            ecdf_axis = axis.twinx()
            ecdf_axis.plot(ecdf_x, ecdf_y, color="darkorange", linewidth=1.5)
            ecdf_axis.set_ylim(0, 1)
            ecdf_axis.set_ylabel("ECDF")
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
