# Changelog

This project follows semantic versioning from ShallowLandslider 1.2 onward. It
also records the main content of every commit currently reachable in the
repository, in reverse chronological order. Earlier releases were not tagged,
so their milestones are identified by commit hash. Descriptions are based on
the committed diffs and messages and summarise the main changes rather than
every edited line.

## Unreleased working tree - multiflow runout and notebook integration

- Added a resumable YAML ensemble launcher that expands dotted parameter grids,
  holds the stochastic seed constant, validates and records every generated
  member, supports process-level parallelism, and writes separate member logs.
- Simplified the distributed YAML to active parameters, disabled unnecessary
  flow routing for its default non-runout soil model, and aligned the execution
  default for DEM smoothing with configuration validation.
- Fixed the piecewise curvature soil-depth relationship so its finite
  steady-state logarithmic branch is reachable, and added physical parameter
  validation plus coverage for all three piecewise regimes.
- Restored runout to the 2025 source-to-endpoint transport concept while using
  Quinn multiple-flow routing. Every selected node above the displacement
  threshold independently starts a branch tree; its original soil column is
  excavated once and divided among terminated endpoints.
- Kept stopped Quinn proportions as terminated branches, normalised transported
  amounts per source, conserved mass, and prevented negative soil depths.
- Made the runout subcomponent own its source nodes, paths, proportions,
  per-source path counts and proportion totals, erosion, deposition, and net
  soil-depth-change diagnostics.
- Added stable runout raster fields and JSON summary metrics. Selected
  initiation, excavated sources, deposition endpoints, runout-only area,
  selected/runout overlap, and combined area are reported separately.
- Required a multiple-flow hill-routing metric for runout. The main drainage
  network may still use D8, but the separate hillslope routing used by runout is
  Quinn by default; single-flow hill routing is rejected.
- Reworked the quick-start notebook into a small, reproducible run that writes
  an analysis-ready run directory, and added an output-analysis notebook that
  reads those results and writes CSV and readable JSON summaries and figures.
- Added regression coverage for multiflow validation, source-to-endpoint branch
  splitting, mass conservation, non-negative soil depth, output footprints,
  JSON summaries, and CLI configuration.

### Scientific behaviour and compatibility

- The core factor-of-safety, critical-acceleration, driving-acceleration, and
  Newmark equations were not changed by the current multiflow/notebook work.
- Intermediate routing nodes are not modified unless they independently qualify
  as sources. A selected node with no valid moving path is retained.
- Runout output from the former segment-by-segment 2026 implementation is not
  directly comparable with the restored source-to-endpoint result.
- The primary analysis output is a run directory rather than a flat CSV/pickle
  pair. Legacy `run.pkl` output remains optional inside that directory.
- The clearer `selected_footprint_percent` summary key retains
  `affected_node_percent` as a backward-compatible alias.

## 2026

### `552c605` - 2026-07-17 - New analysis tools

- Introduced version 1.2 metadata and a versioned run-directory output schema.
- Added manifests, JSON run summaries, explicit region tables, optional
  Parquet/Zarr output, and NumPy raster fallbacks.
- Added the `analysis` package and command-line output-analysis entry point.
- Expanded configuration validation, provenance capture, tests, requirements,
  README documentation, and the release-oriented changelog.
- Made `regions.csv` the always-available region output, with Parquet and Zarr
  dependent on optional libraries and memory-mappable NumPy rasters as the
  dependency-safe fallback.
- Existing flat-output consumers should migrate to `analysis.load_run()` or
  `analysis.load_region_ensemble()`.

### `78adf99` - 2026-07-17 - Curvature and large-DEM fixes

- Reworked curvature-derived soil depth for overlapped chunk processing and
  stitched tile cores into the global result.
- Preserved nodata masks, padded NaN-heavy edge tiles for curvature evaluation,
  skipped all-nodata tiles, and enforced sufficient overlap.
- Forwarded curvature configuration consistently and respected the configured
  maximum soil depth.
- Added CLI regression tests for chunked curvature and edge/nodata cases.
- Derived the minimum tile overlap from the curvature stencil and local-
  standard-deviation window, and handled constant-curvature and all-nodata
  tiles without terminating the run.
- Reused stitched finite tile cores when building the global grid instead of
  recomputing curvature-derived soil depth over the full DEM.
- These fixes can change results near nodata boundaries, with non-default soil-
  depth maxima, and in chunked curvature runs. The underlying stability and
  Newmark equations were unchanged.

### `25187e9` - 2026-06-02 - Remove derived measured-data cache

- Removed `utils/measured_data.pkl`; measured-data caches became generated
  artefacts rather than repository source files.

### `8dd4a92` - 2026-06-02 - Restore package initializer

- Added the root `__init__.py` omitted from the preceding standalone-release
  restructuring.

### `8c47f1d` - 2026-06-02 - Prepare standalone release

- Moved the main and runout components into the current
  `components/shallow_landslider` package and added package initializers.
- Revised the component, runout implementation, CLI, YAML configuration,
  quick-start notebook, utilities, tests, README, and ignore rules.
- Added bundled Nepal and New Zealand example DEMs for reproducible runs.
- Established the repository layout used by the current standalone component.

### `d3bc9c6` - 2026-04-17 - Logging and HPC processing

- Added structured progress logging and revised the CLI for smoother HPC
  execution.
- Refactored and optimised component processing, including staged execution and
  reduced intermediate overhead.
- Updated the quick-start notebook and utilities to match the new workflow.

### `a389072` - 2026-04-09 - README corrections

- Corrected and simplified documentation following the runout/CLI integration.

### `9f5d887` - 2026-04-09 - Runout, CLI, configuration, and datasets

- Added the dedicated runout subcomponent and connected soil erosion/deposition
  updates to Newmark displacement.
- Added the first YAML-driven CLI, example configuration, and environment file.
- Added measured-landslide inventories and zonal statistics for Japan, Nepal,
  New Zealand, and Papua New Guinea.
- Substantially revised utilities, quick-start usage, region geometry and
  splitting calculations, slope caching, and soil-depth handling.
- Changed the factor-of-safety soil-depth array to `float32` at that point in
  history and changed its depth guard from replacing exactly zero to clamping
  depths below 0.001 m.
- Removed the older standalone simulator trial and moved workflows toward the
  component plus CLI architecture.

### `1d21981` - 2026-02-10 - Test-suite update

- Reworked component, stability/displacement, region-selection/splitting, and
  utility tests to match the refactored implementation.

### `d9dfbfc` - 2026-02-10 - Pre-merge polish

- Applied small fixes to the trial runner, component, utility exports, and
  utility implementation before merging the refactored work.

### `7d6a74c` - 2026-01-09 - Remove integration debug code

- Removed temporary debugging and demonstration code left in the integrated
  component.

### `f7dcb63` - 2026-01-09 - Quick-start measured-data cache

- Added an example derived measured-data pickle for the original quick-start
  workflow. This cache was later removed by `25187e9`.

### `2679719` - 2026-01-09 - Integrate helpers into the component

- Consolidated stability, displacement, region, selection, and split helpers
  into the main component.
- Moved general-purpose helpers into the current `utils` package.
- Added the first `ShallowLandslider_quickstart.ipynb` and updated the working
  simulator script.

### `8c6e885` - 2026-01-08 - Merge pull request 4

- Merged the final-release branch containing Arc/ASCII DEM input support and
  the updated utilities/working script. The merge message noted possible NaN
  edge behaviour, addressed later in the curvature/chunking work.

### `77294e7` - 2026-01-08 - Arc DEM utility updates

- Updated utilities and the working simulator to accept Arc-output DEMs.
- Revised dependency constraints and DEM/nodata handling in the example flow.

### `6f98173` - 2026-01-07 - Remove superseded implementations

- Deleted the old `auxiliary_functions`, `data_analysis`, legacy scripts, and
  earlier class variants after the helper/component refactor.

### `6008942` - 2026-01-06 - Full testing suite

- Added analytic and integration tests for the component, stability,
  displacement, region selection/splitting, and utilities.
- Corrected friction-angle conversion before scientific calculations, changed
  the default submerged-soil proportion to 0.5, and improved selection and NaN
  handling while making the code testable.
- Added package exports and updated requirements and the simulator trial.

### `f4da8ab` - 2026-01-06 - Repository data cleanup

- Removed tracked DEM rasters and large derived measured-data pickle files from
  the repository.

### `7d2099e` - 2026-01-06 - Refactored helpers and README

- Introduced the `helper_functions` package for displacement, stability,
  regions, selection, splitting, and shared utilities.
- Expanded the README and revised the component and simulator to use the new
  helper layout.
- Removed the obsolete first-generation test module pending the new suite.

### `aa1e19c` - 2026-01-03 - Landlab component update

- Added a dedicated `shallow_landslide_component.py` and a simulator trial,
  beginning the transition from experimental classes to a Landlab component.

## 2025

### `e94761c` - 2025-12-02 - Merge excess-topography work

- Merged the `excess_topo_calculation` branch before the final component
  refactor; the merge itself introduced no separate file changes.

### `596019f` - 2025-12-02 - Soil-depth and analysis expansion

- Expanded soil-depth generation beyond uniform/elevation relationships to
  curvature, drainage-area, and composite parameterisations.
- Added global and local curvature-standard-deviation relationships and
  extensively revised terrain, region, selection, Newmark, and analysis code.
- Added an eastern measured-data cache and updated trial workflows.

### `ad2f9e0` - 2025-09-29 - Data-analysis package

- Split statistical analysis into a new `data_analysis` package.
- Expanded statistical helpers and refactored the main/trial classes and model
  analysis script around those tools.

### `b225a69` - 2025-09-12 - Dataset statistics

- Added a model-data analysis script and statistical comparison workflows.
- Expanded topographic and terrain analysis and added a southern measured-data
  cache.

### `5b4fbf1` - 2025-08-28 - Measured-data pickling

- Added loading/building of cached measured-landslide data and revised
  selection and terrain helpers to use it.
- Added an experimental soil-depth model script and the first tracked
  measured-data pickle.

### `5e493e6` - 2025-08-12 - Untrack IDE files

- Removed remaining tracked Spyder project configuration files after adding
  ignore rules.

### `6c1e384` - 2025-08-12 - Add ignore rules

- Added the initial `.gitignore` for generated and local-development files.

### `1f17f58` - 2025-08-12 - Remove generated files

- Removed tracked Spyder configuration, Python bytecode, and cache directories.

### `ec6ced4` - 2025-08-12 - Pre-Windows update

- Added the first requirements file and broadly revised I/O, simulation,
  terrain, statistics, topographic helpers, and trial scripts.
- Included portability and workflow adjustments made before moving development
  to Windows 11.

### `cb0754f` - 2025-07-07 - Topographic helper branch

- Added a dedicated topographic-functions module and exposed it from the
  auxiliary package.
- Updated the trial workflow to use the new helpers.

### `e7b489d` - 2025-06-25 - Merge pull request 2

- Merged the `add_region_splitting` branch; the merge itself introduced no
  separate changes beyond the commits below.

### `1a70861` - 2025-06-25 - Region splitting milestone

- Completed the first region-splitting workflow and integrated it into the
  shallow-landslider class and trial script.
- Revised region/statistical/I/O helpers and added the first substantial test
  module.

### `3b8aa75` - 2025-06-19 - Splitting fixes

- Applied minor corrections to the main and trial classes after initial region
  splitting was introduced.

### `6eabe58` - 2025-06-17 - Initial working splitting

- Added the first working statistical region-splitting implementation and
  connected it to the main/trial classes.
- Marked the implementation as needing further testing, which followed in
  `1a70861`.

### `63f3317` - 2025-06-16 - Merge pull request 1

- Merged the then-current master branch; the merge commit contains no separate
  file changes.

### `79303e0` - 2025-06-16 - Intermediate plotting and experiments

- Expanded intermediate plotting and region/statistical helpers.
- Added several experimental class variants, an inverse-gamma script, example
  DEMs, and development artefacts used during early exploration.

### `aaa8d6f` - 2025-06-10 - Split auxiliary functions into modules

- Replaced the single large `auxiliary_functions.py` with modules for I/O,
  Newmark calculations, regions, selection, simulation, statistics, and
  terrain processing.
- Updated the main class to use the modular helper package.

### `a02e08f` - 2025-06-04 - Intermediate plotting

- Added intermediate diagnostic plotting and revised the main and trial classes
  and the original auxiliary-function collection.

### `a31114e` - 2025-05-15 - First shallow-landslider implementation

- Added the original experimental class, trial script, and comprehensive
  auxiliary-function module.
- Introduced the first factor-of-safety, transient critical acceleration,
  driving acceleration, region selection, Newmark displacement, and early
  source-to-endpoint runout/path implementations.

### `b0856cc` - 2025-05-08 - Initial repository

- Created the repository with the GNU GPLv3 licence and an initial README.
