# HPC workflow

The tracked files in this directory run Nepal and Japan single simulations,
Cartesian ensembles, and output analysis from the main ShallowLandslider source
tree. Model code is not duplicated here.

Large inputs and generated results live outside Git. The ignored
`hpc/workspace` symbolic link gives the tracked configurations a stable path on
both a workstation and the cluster.

## Set up the data workspace

Run the setup script once on each machine, using an absolute path on suitable
project or scratch storage:

```bash
./hpc/setup_workspace.sh /scratch/$USER/EQShallowLandslider_data
```

This creates:

```text
EQShallowLandslider_data/
├── input_data/
│   ├── dem/
│   ├── nepal/
│   ├── japan/
│   └── nz/
├── runs/
└── analysis_output/
```

Place the production DEMs in `input_data/dem/` as:

- `gorkhadem_05g.asc`
- `iwate_dem_utm_clip_50gbox.asc`
- `Kaikoura_04g_clipped_DEM.asc`

The Nepal, Japan, and New Zealand measured inventory files belong in their
corresponding `input_data/<region>/` directories (`nz` for Kaikoura). They are
used for width splitting and model-observation analysis.

The workspace link, runs, analysis products, DEMs, derived pickles, and Slurm
logs are intentionally excluded from Git.

## Environment and validation

The submission scripts expect the `landlab_dev` Conda environment and the
Plymouth `cpu_shared` partition. Review the account, partition, memory, CPU, and
walltime directives before submitting elsewhere.

Submit jobs from the repository root. For local validation, retain the
repository path before entering the data workspace so relative configuration
paths resolve exactly as they do in Slurm:

```bash
PROJECT_DIR="$PWD"
cd hpc/workspace

python "$PROJECT_DIR/run_landslide_model_cli.py" \
  --config "$PROJECT_DIR/hpc/configs/nepal_config.yaml" --validate-only
python "$PROJECT_DIR/run_landslide_ensemble.py" \
  --config "$PROJECT_DIR/hpc/configs/nepal_config.yaml" --dry-run

python "$PROJECT_DIR/run_landslide_model_cli.py" \
  --config "$PROJECT_DIR/hpc/configs/japan_config.yaml" --validate-only
python "$PROJECT_DIR/run_landslide_ensemble.py" \
  --config "$PROJECT_DIR/hpc/configs/japan_config.yaml" --dry-run

python "$PROJECT_DIR/run_landslide_model_cli.py" \
  --config "$PROJECT_DIR/hpc/configs/kaikoura_config.yaml" --validate-only
python "$PROJECT_DIR/run_landslide_ensemble.py" \
  --config "$PROJECT_DIR/hpc/configs/kaikoura_config.yaml" --dry-run

cd "$PROJECT_DIR"
```

## Submit model jobs

Nepal is the default configuration:

```bash
sbatch hpc/slurm/run_single.sbatch
sbatch hpc/slurm/run_ensemble.sbatch
```

Slurm executes a spooled copy of each batch script, so the launchers locate the
repository through `SLURM_SUBMIT_DIR`. Submit from the repository root as shown
above. When submitting from elsewhere, provide the checkout explicitly:

```bash
sbatch --export=ALL,EQSL_PROJECT_DIR=/absolute/path/to/EQShallowLandslider \
  /absolute/path/to/EQShallowLandslider/hpc/slurm/run_ensemble.sbatch
```

Select Japan by supplying its tracked configuration:

```bash
sbatch --export=ALL,CONFIG="$PWD/hpc/configs/japan_config.yaml" hpc/slurm/run_single.sbatch
sbatch --export=ALL,CONFIG="$PWD/hpc/configs/japan_config.yaml" hpc/slurm/run_ensemble.sbatch
```

For Kaikoura, submit the single run first and inspect it before launching the
45-member ensemble:

```bash
sbatch --export=ALL,CONFIG="$PWD/hpc/configs/kaikoura_config.yaml" hpc/slurm/run_single.sbatch
sbatch --export=ALL,CONFIG="$PWD/hpc/configs/kaikoura_config.yaml" hpc/slurm/run_ensemble.sbatch
```

The measured Kaikoura ShakeMap configuration uses the calibrated ensemble
parameters and a float32 NumPy PGA raster to avoid repeatedly parsing the much
larger long-decimal ASCII export:

```bash
sbatch --export=ALL,CONFIG="$PWD/hpc/configs/kaikoura_shakemap_best_fit.yaml" hpc/slurm/run_single.sbatch
```

The raster-PGA interface is region-independent. A future Gorkha configuration
can set `pga.distribution: raster`, point `pga.horizontal_path` at an aligned
ASCII or `.npy` grid, and either provide `pga.vertical_path` or set
`pga.vertical_to_horizontal_ratio`. NumPy grids do not contain georeferencing,
so verify and resample the source against the DEM before conversion. Set
`pga.row_order: north_to_south` when the first NumPy row is the northern edge,
as in an array copied directly from ESRI ASCII; use `south_to_north` for an
array already arranged in Landlab node order. The model enforces an identical
array shape and masks PGA wherever the DEM is nodata. For a spatial preflight,
enable `pga.alignment_check`, specify the raster's background or nodata value,
and set a minimum mask intersection-over-union. The run aborts before stability
calculation if the oriented raster footprint does not sufficiently match the
DEM nodata footprint. Keep these event rasters in
`hpc/workspace/input_data/<region>/` and transfer them separately from Git.

The ensemble launcher runs one member at a time. Increase `--jobs`, requested
CPUs, and memory together only after measuring one complete member.

## Analyse completed runs

Nepal is again the default region:

```bash
sbatch hpc/slurm/analyse_single.sbatch
sbatch hpc/slurm/analyse_ensemble.sbatch
```

For Japan:

```bash
sbatch --export=ALL,REGION=japan hpc/slurm/analyse_single.sbatch
sbatch --export=ALL,REGION=japan hpc/slurm/analyse_ensemble.sbatch
```

Select a named single-run experiment with `RUN_NAME`. The launcher reads from
`runs/<region>/<run-name>` and writes to
`analysis_output/<region>_<run-name>`:

```bash
sbatch --export=ALL,REGION=kaikoura,OBSERVED_REGION=nz,\
RUN_NAME=shakemap_elevation_linear_corrected hpc/slurm/analyse_single.sbatch
```

Ensemble analysis automatically creates controlled sensitivity plots for every
swept parameter in addition to the per-run distribution and spatial plots.
The ensemble and synthetic analysis launchers request four CPUs and use four
per-run analysis workers by default. Override this with, for example,
`--export=ALL,ANALYSIS_JOBS=2` when raster size or node memory calls for fewer
concurrent workers.

## Transfer data separately from code

Git moves code, configs, and Slurm scripts. Use `rsync` for inputs and selected
results:

```bash
# Workstation to HPC
rsync -av EQShallowLandslider_data/input_data/ \
  hpc-host:/scratch/$USER/EQShallowLandslider_data/input_data/

# HPC to workstation
rsync -av hpc-host:/scratch/$USER/EQShallowLandslider_data/analysis_output/ \
  EQShallowLandslider_data/analysis_output/
```

Large raster bundles can remain on the cluster while summaries, region tables,
and figures are copied back selectively.

## Generate synthetic terrain ensembles

The synthetic generator remains in the main source tree; the Slurm launcher
executes it from there and writes all large products through `hpc/workspace`.
The `_HPC` or scratch workspace therefore contains data and results, not a
second copy of the application code.

The default array uses the bundled landscape seed and pairs it across the
SPACE-only and weathering–Taylor regolith models. Refinement factor 4 matches
the bundled landscape's 120 m evolution grid and 30 m output grid:

| Task | Seed | Regolith model |
|---:|---:|---|
| 0 | 20260804 | `space` |
| 1 | 20260804 | `weathering_taylor` |

Submit it from the main repository root:

```bash
sbatch hpc/slurm/generate_synthetic_topography.sbatch
```

Outputs are written beneath
`hpc/workspace/input_data/dem/synthetic_ensemble/`. Each realization contains
a topographic-elevation raster, matching soil-depth and bedrock-elevation
rasters, an uplift-rate raster, a four-panel elevation/soil/slope/drainage PNG,
and JSON provenance.

Override the seeds and array range together when expanding the experiment.
There must be two tasks per seed; the optional percent suffix limits concurrent
jobs:

```bash
SYNTHETIC_SEEDS="41 42 43 44 45" \
sbatch --array=0-9%3 hpc/slurm/generate_synthetic_topography.sbatch
```

Principal generator settings can also be exported at submission, including
`REFINEMENT_FACTOR`, `ITERATIONS`, `UPLIFT_RATE`, `UPLIFT_BACKGROUND_RATE`,
`UPLIFT_PATTERN`, `UPLIFT_CENTER_X_FRACTION`, `UPLIFT_CENTER_Y_FRACTION`,
`UPLIFT_SIGMA_X_FRACTION`, `UPLIFT_SIGMA_Y_FRACTION`,
`SOIL_PRODUCTION_MAXIMUM_RATE`,
`SOIL_PRODUCTION_DECAY_DEPTH`, `SOIL_TRANSPORT_VELOCITY`,
`SOIL_TRANSPORT_DECAY_DEPTH`, and `CRITICAL_SLOPE`.

Use the elevation and soil file from the same array task in a landslide
configuration. Paths remain relative to the data workspace because the Slurm
model launchers run from `hpc/workspace`:

```yaml
dem_path: "input_data/dem/synthetic_ensemble/synthetic_space_seed20260804_600x800_30m.asc"

soil_params:
  distribution: "raster"
  soil_depth_path: "input_data/dem/synthetic_ensemble/synthetic_space_seed20260804_600x800_30m_soil_depth.asc"
  cohesion_eff: 15000
  angle_int_frict: 30
  submerged_soil_proportion: 0.5
```
