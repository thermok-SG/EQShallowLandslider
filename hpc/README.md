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
│   └── japan/
├── runs/
└── analysis_output/
```

Place the production DEMs in `input_data/dem/` as:

- `gorkhadem_05g.asc`
- `iwate_dem_utm_clip_50gbox.asc`

The Nepal and Japan measured inventory files belong in their corresponding
`input_data/<region>/` directories. They are used for width splitting and
model-observation analysis.

The workspace link, runs, analysis products, DEMs, derived pickles, and Slurm
logs are intentionally excluded from Git.

## Environment and validation

The submission scripts expect the `landlab_dev` Conda environment and the
Plymouth `cpu_shared` partition. Review the account, partition, memory, CPU, and
walltime directives before submitting elsewhere.

Run commands from the repository root. Validate both region configurations
without allocating a model job:

```bash
python run_landslide_model_cli.py --config hpc/configs/nepal_config.yaml --validate-only
python run_landslide_ensemble.py --config hpc/configs/nepal_config.yaml --dry-run

python run_landslide_model_cli.py --config hpc/configs/japan_config.yaml --validate-only
python run_landslide_ensemble.py --config hpc/configs/japan_config.yaml --dry-run
```

## Submit model jobs

Nepal is the default configuration:

```bash
sbatch hpc/slurm/run_single.sbatch
sbatch hpc/slurm/run_ensemble.sbatch
```

Select Japan by supplying its tracked configuration:

```bash
sbatch --export=ALL,CONFIG="$PWD/hpc/configs/japan_config.yaml" hpc/slurm/run_single.sbatch
sbatch --export=ALL,CONFIG="$PWD/hpc/configs/japan_config.yaml" hpc/slurm/run_ensemble.sbatch
```

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

Ensemble analysis automatically creates controlled sensitivity plots for every
swept parameter in addition to the per-run distribution and spatial plots.

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
