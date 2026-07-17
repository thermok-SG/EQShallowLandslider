# Changelog

This project follows semantic versioning from ShallowLandslider 1.2 onward.
The repository did not contain Git release tags before version 1.2, so earlier
milestones are referenced by commit hash.

## 1.2.0 - In development (2026-07-17)

Release status: not yet tagged. Development branch:
`fix/curvature-regolith-chunking`.

### Added

- Versioned, unique output directories for every CLI run.
- A JSON manifest recording:
  - model and output-schema versions;
  - complete model configuration;
  - Git commit, branch, exact tag when available, and dirty status;
  - Python and scientific-package versions;
  - grid shape, spacing, DEM details, execution mode, and runtime;
  - an inventory of output files.
- A compact `summary.json` containing candidate/selected region counts,
  affected-node percentage, selected area statistics, and selected proportion.
- Label-preserving region output in CSV and, when `pyarrow` is installed,
  Parquet format.
- Per-region analysis fields including cell count, centroid, mean factor of
  safety, mean critical acceleration, mean acceleration difference, and maximum
  Newmark displacement.
- Chunked Zarr raster output when `xarray` and `zarr` are installed.
- A dependency-safe fallback of one memory-mappable NumPy file per raster.
- Analysis utilities for discovering runs, loading one run lazily, combining
  ensemble region tables, and plotting histograms with ECDF overlays.
- The `analyse_landslide_outputs.py` command-line analysis entry point.
- Regression tests for the v1.2 output schema, analysis loading, and plotting.

### Curvature and large-DEM fixes

The following changes were first committed in `78adf99` on 2026-07-16
(`sg/maybe fixed curvature issue for edge chunks of large DEMs`). No Git tag is
associated with that commit.

- Curvature-derived soil depth is calculated within overlapped tiles and the
  tile-core soil depths are stitched into the global result. The CLI no longer
  performs a second full-DEM RichDEM curvature calculation after chunking.
- The minimum overlap is derived from the curvature stencil and local-standard-
  deviation window. For a window of five cells, the required overlap is three
  cells.
- Configured curvature options such as `window` and `scale` are forwarded to the
  soil-depth calculation.
- Constant-curvature tiles no longer raise `ValueError`; the configured
  relationship is evaluated with zero curvature variance.
- Curvature relationships now clip against the `max_soil_depth` argument rather
  than an independent hard-coded 1.5 m fallback.
- Composite elevation-curvature soil depth now uses the configured curvature
  relationship and its options.
- DEM nodata masks are retained during chunking.
- NaN-heavy edge tiles are temporarily padded from their nearest valid elevation
  before RichDEM curvature calculation. The original nodata cells remain closed,
  receive zero soil depth, and cannot be classified as unstable.
- Tiles containing only nodata are skipped.
- The finite, padded tile cores are reused when constructing the global grid so
  edge NaNs do not re-enter downstream calculations.

### CLI and configuration fixes

- Added complete YAML validation and a `--validate-only` CLI option.
- Added explicit `chunking.mode` values (`auto`, `always`, and `never`) and a
  matching command-line override; the older `enable_auto` key remains accepted.
- Added YAML controls for PGA centre, random centre, PGA seed, and component
  worker count.
- Non-uniform PGA distributions are now generated once in global coordinates
  and sliced into tiles. Previously, every tile independently changed the PGA
  centre and distance normalization.
- Chunked runs now stitch and retain critical and driving acceleration in
  addition to factor of safety and acceleration difference. These arrays are
  also supplied to global candidate selection and v1.2 raster/region outputs.
- Stitched soil depth is retained as the `float64` dtype required by Landlab;
  the former `float32` global field stopped chunked runs before selection.
- The configured submerged-soil proportion is now forwarded to the static
  factor-of-safety calculation. Critical acceleration already used this value.
- Unsupported chunked combinations (runout and drainage-area soil depth) now
  stop with an actionable error rather than being ignored or failing later.
- Runout flag dependencies, soil/PGA methods, numerical ranges, tile geometry,
  Zarr chunks, and selection options are validated before model execution.
- Zarr output explicitly writes the version 3 format supported by the v1.2
  dependency range.

### Scientific-output implications

- Core factor-of-safety, critical-acceleration, driving-acceleration, and Newmark
  equations were not changed by commit `78adf99`.
- Results can change near nodata boundaries because curvature is now evaluated
  from nearest-valid padding rather than propagating NaNs or failing.
- Curvature runs with a non-default maximum soil depth can change because the
  configured maximum is now applied correctly.
- Chunked curvature results can change because the globally reconstructed model
  now uses the same stitched soil depths that produced tile stability, rather
  than independently recomputing soil depth on the full DEM.
- Constant-curvature tiles now produce results where previous runs terminated.
- Factor-of-safety results change when `submerged_soil_proportion` differs from
  0.5 because the configured value is now used instead of that equation's
  default. The underlying equation itself is unchanged.
- Chunked runs using a spatial PGA distribution now use one continuous global
  forcing field rather than a separately normalized distribution in each tile.

### Output compatibility

- The primary output is now a run directory rather than a flat CSV/pickle pair.
- `regions.csv` is always written. `regions.parquet` and `rasters.zarr` depend on
  their optional storage libraries.
- Legacy pickle output remains available through `save_pickle: true`, but the
  pickle is stored as `run.pkl` inside the run directory.
- Existing flat-output consumers should migrate to `analysis.load_run()` or
  `analysis.load_region_ensemble()`.

## Earlier unversioned milestones

These entries document important scientific changes identified from Git history;
they are not retroactive semantic-version releases.

### `8c47f1d` - 2026-06-02

- Prepared the standalone component layout used by the current repository.
- Consolidated selection, splitting, stability, displacement, and runout code
  beneath `components/shallow_landslider`.

### `9f5d887` - 2026-04-09

- Added the runout subcomponent and soil erosion/deposition updates.
- Cached slope arrays and changed factor-of-safety soil depth to `float32`.
- Changed the factor-of-safety depth guard from replacing exactly zero depth to
  clamping all depths below 0.001 m.
- Revised region geometry and slope-direction dimension calculations.

### `6008942` - 2026-01-06

- Converted the component friction-angle input from degrees to radians before
  scientific calculations.
- Changed the default submerged-soil proportion from 0.0 to 0.5.
- Revised probabilistic/PGA-weighted selection behavior and NaN handling.
- Added analytic regression tests for stability and displacement equations.

### `596019f` - 2025-12-02

- Expanded soil depth beyond uniform/elevation relationships to curvature,
  drainage-area, and composite parameterisations.
- Introduced global and local curvature-standard-deviation relationships.

### `a31114e` - 2025-05-15

- Added the original factor-of-safety, transient critical acceleration, driving
  acceleration, and Newmark displacement implementations.
