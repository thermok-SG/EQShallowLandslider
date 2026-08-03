# **ShallowLandslider: A Landlab Component for Earthquake-Induced Landslide Simulation**

## **Overview**
`ShallowLandslider` is a _component_ designed to model the distribution of shallow landslides triggered by seismic shaking. It integrates physics-based stability calculations, probabilistic selection methods, and optional width-based recursive splitting to simulate landslide occurrence and distribution across a DEM.

This tool is intended for **geomorphologists, hazard modellers, and Earth scientists** who need reproducible, scalable landslide simulations. Its Landlab-based architecture means that it can be integrated within larger landscape evolution modelling scenarios.

---

## **Key Features**
- **Factor of Safety (FoS)** and **critical acceleration** calculations for shallow landslides.
- **Probabilistic selection** and **PGA-weighted selection** workflows.
- **Recursive width-based splitting** using KDE-informed thresholds.
- **Region property extraction** (area, slope, aspect, relief, shape metrics).

## Optional features/utilities
- **Easy DEM download from OpenTopography** using BMI-topography makes it easy to quickly test scenarios
- **Flexible soil depth assignment** (uniform, elevation-based, curvature-based, drainage-area-based) to test the effect of different soil depth distributions
- **Earthquake PGA grid** generator that can handle multiple spatial distributions to test the effect of different seismic scenarios
- **Optional multiflow runout routing** that divides each failed source node's
  soil among Quinn hill-flow branches and deposits it at their endpoints.
- **Analysis-ready outputs** with JSON provenance and summaries, region tables,
  memory-mappable rasters, and separate selected/runout footprints.

## **Installation**
### **Requirements**
- Python ≥ 3.9
- [Landlab](https://landlab.readthedocs.io/)
- NumPy, Pandas, SciPy, scikit-image

Create the provided environment:

```bash
conda env create -f environment.yml
conda activate shallow_landslider
```

Or install the Python dependencies into an existing environment:

```bash
pip install -r requirements.txt
```

## **Measured Data & KDE Bundle (Recommended)**
Many workflows **benefit strongly** from measured landslide data to guide *statistically based region splitting* and to compare modelled vs observed characteristics. While technically optional, this step is **highly recommended** for realistic run outcomes.

For physics-only runs, omit `split_by_width_config` or set it to `None`. The component will still compute stability, unstable regions, aspect subgroups, candidate selection, and optional displacement; it simply skips KDE-informed width splitting.

### What this provides
- A **preprocessed bundle** (a single pickle) containing:
  - `measured_data` (e.g., length/width records)
  - `measured_spatial_stats` and variants (zonal stats like elevation, slope, aspect)
  - A **bivariate KDE** (`kde_data`) over `(length_m, width_m)` with its `kde_transform` metadata
- Used by the recursive splitting workflow to split overly wide regions in a way that is consistent with observations.

### How it works
Use `pickle_or_not_to_pickle(...)` to *load an existing pickle* (fast), or *build one from CSV* the first time and cache it:

- The first build may take a few minutes depending on dataset size. Subsequent runs read the pickle in milliseconds. 

### Data requirements
- Keep the **CSV files** in version control. The `measured_data.pkl` files are derived caches and can be regenerated from the CSVs when the notebook or CLI runs.
- **CSV columns** for KDE fitting: `length_m`, `width_m`. Values must be **positive** if log-transform is enabled (default).
- If you enable grouping (`category_col`), ensure that each category has **≥ 5 samples**—sparser groups are skipped.
- The function `fit_bivariate_kde(...)` internally called by `pickle_or_not_to_pickle` uses `scipy.stats.gaussian_kde` and supports optional log-transforms and per-category bandwidths.

Recommended CSV structure:

| File | Purpose | Required columns | Common optional columns |
|---|---|---|---|
| `measuredLandslides_all.csv` | Landslide inventory used to fit length-width KDEs | `length_m`, `width_m` | `area_m2`, `ID`, `name` |
| `measuredLandslides_all_ZonalStats.csv` | Observed landslide terrain statistics used for comparison plots and filtering | `Area_m2` or `area_m2` | `Elevation_m_mean` or `Elevation_mean`, `Slope_deg_mean`, `Aspect_deg_median` |
| Region-clipped zonal stats, optional | Subregion-specific observed comparison data | `Area_m2` or `area_m2` | `Elevation_m_mean`, `Slope_deg_mean`, `Aspect_deg_median` |

### Troubleshooting
- `ValueError: Cannot log-transform ... contains values <= 0` → turn off log-transform or filter/clean your data.
- If your CSV column names differ, pass the correct `x_col`/`y_col` into `fit_bivariate_kde(...)` when you build the pickle the first time.
- Use `verbose=True` to see a concise progress log while building/loading the pickle.

## **Optional Runout**

Runout is available as an optional soil-depth update after selected landslides
have Newmark displacement values. The grid must first be routed with
`PriorityFloodFlowRouter` using `separate_hill_flow=True` and a multiple-flow
hill metric such as `hill_flow_metric="Quinn"`.

The router exposes two distinct routing systems:

- `flow_metric="D8"` may still be used for the main drainage network.
- `hill_flow_metric="Quinn"` creates the multiple-receiver proportions used by
  sediment runout. The runout component does not route sediment with the main
  D8 receiver field.

Single-receiver hill routing is rejected when runout is enabled because it
cannot divide material among branching sediment paths.

### Source and endpoint behaviour

The selected and runout footprints have related but different meanings:

- A **selected node** has `landslide__selected_labels > 0`.
- A **runout source** is a selected node whose finite Newmark displacement is
  greater than `displacement_threshold`.
- Every runout source independently starts its own Quinn branch tree. A node is
  not treated as one bulk source merely because it belongs to a selected
  region.
- The source's original soil column is divided among its terminated branch
  proportions and excavated exactly once.
- Material is deposited only at branch endpoints. An intermediate routing node
  is not eroded or deposited upon merely because another path crosses it; it
  can nevertheless be excavated independently if it is also a runout source.
- A source with no valid moving path is retained rather than artificially
  excavated.
- Stopped Quinn proportions are retained as terminated branches. Per-source
  allocation is mass conserving and does not allow negative soil depth.

Runout is only executed when all three component flags are enabled:

- `compute_displacement=True`
- `enable_runout=True`
- `update_soil=True`

In the YAML CLI config, this means using `chunking.mode: never`, keeping
`flow_params.enable` and `flow_params.separate_hill_flow` true, and setting
`simulation.compute_displacement`, `simulation.enable_runout`, and
`simulation.update_soil` to true. Runout modifies `soil__depth` in place and
caches the source nodes, paths, proportions, per-source branch totals, and
erosion/deposition arrays on the runout subcomponent.

The latest runout diagnostics are available at `ls.results["runout"]`:

| Key | Meaning |
|---|---|
| `failed_nodes` | Selected nodes above the displacement threshold |
| `paths` | Terminated multiflow paths, including their source and endpoint |
| `path_proportions` | Quinn branch weight corresponding to each path |
| `path_details` | Paths and weights grouped by initiating source |
| `source_proportion_sums` | Sum of terminated branch weights per source |
| `source_path_counts` | Number of terminated branches per source |
| `erosion` | Soil thickness removed at source nodes in the latest step |
| `deposition` | Soil thickness deposited at endpoints in the latest step |
| `soil_depth_change` | Net deposition minus erosion at every node |

## **Quick start**
```python
from components.shallow_landslider import ShallowLandslider

from utils import (
    get_topo,
    apply_soil_depth,
    pickle_or_not_to_pickle,
    calculate_terrain_attribute,
    generate_acceleration_grid,
)

from landlab.components import PriorityFloodFlowRouter

# 0. Load measured data and build the KDE cache from CSVs.
file_name_dict = {
    "file1": "input_data/nepal/measuredLandslides_all.csv",
    "file2": "input_data/nepal/measuredLandslides_all_ZonalStats.csv",
}

bundle = pickle_or_not_to_pickle(
    file_name_dict,
    pickle_path="input_data/nepal/measured_data.pkl",
    min_area=900,
    verbose=True,
)

# Access KDE for recursive split
kde_dict = {
    "kde_data": bundle["kde_data"],
    "kde_transform": bundle["kde_transform"],
}

# 1. Build a Landlab grid from the bundled DEM.
grid, z, nodata_mask = get_topo(
    buffer=0.01,
    dem_type="SRTMGL1",
    load_dem="input_data/dem/SRTMGL1_28.169999999999998_85.03_28.3_85.21000000000001.asc",
)

# 2. Initialize and run flow router
pf = PriorityFloodFlowRouter(
    grid,
    flow_metric="D8",
    separate_hill_flow=True,  # required for optional runout
    hill_flow_metric="Quinn",  # multiflow proportions split runout paths
    depression_handler="fill",
    update_hill_depressions=True,
    accumulate_flow=True,
)
pf.run_one_step()

# 3. Apply soil depth
curv = calculate_terrain_attribute(
    grid=grid,
    field_name="topographic__elevation",
    attrib="planform_curvature",
)
soil_depth = apply_soil_depth(grid, max_soil_depth=1.5, distribution="curvature")

# 4. Generate earthquake PGA grids
pga_h, pga_v = generate_acceleration_grid(grid, horizontal_max=0.6, vertical_max=0.2)

# 5. Instantiate and run the component.
ls = ShallowLandslider(
    grid,
    cohesion_eff=20e3, # Pa
    angle_int_frict=30.0,  # degrees
    submerged_soil_proportion=0.5,
    pga_h=pga_h,
    pga_v=pga_v,
    selection_method="probabilistic",  # or "pga_weighted"
    proportion_method="conservative",
    random_seed=5000,
    compute_displacement=True,
    time_shaking=5.0,
    displacement_threshold=0.0,
    enable_runout=True,
    update_soil=True,
    verbose=True,  # optional
    split_by_width_config={
        "kde_data": kde_dict["kde_data"],
        "kde_transform": kde_dict["kde_transform"],
        "convergence_threshold": 0.75,
        "min_region_size": 10, # minimum number of pixels a region must have to be a split candidate
        "max_iterations": 10, # max number of recursive iterations to prevent infinite loops
        "width_threshold": 1.5, # Ratio of actual width / expected width beyond which a region is flagged for splitting
    }
)

ls.run_one_step()

# 6. Access results
print(ls.results["group_properties"].head())
print(ls.results["runout"]["source_path_counts"])
```

For a faster first run, use
[`ShallowLandslider_quickstart.ipynb`](ShallowLandslider_quickstart.ipynb). It
crops the bundled DEM to a small domain, enables Quinn runout, and writes an
analysis-ready directory beneath `runs/`.

## CLI outputs and analysis (v1.2)

Validate a YAML file without loading its DEM or starting a model run:

```bash
python run_landslide_model_cli.py \
  --config ShallowLandslider_config.yaml \
  --validate-only
```

`chunking.mode` accepts `auto`, `always`, or `never`; the command-line
`--chunking` option can override it for a particular job. Curvature soil models
automatically increase an undersized tile overlap. Drainage-area soil depth and
runout require global flow-routing fields and therefore fail early if chunked
execution is selected. A non-null `simulation.custom_proportion` overrides
`proportion_method`; leave it as `null` to use the named method.

Each CLI execution writes a unique, self-describing directory beneath
`output_dir`:

```text
runs/<timestamp>_<parameter-name>/
├── manifest.json
├── summary.json
├── regions.csv
├── regions.parquet       # when pyarrow is available
├── rasters.zarr/         # when xarray/zarr are available
└── rasters/              # .npy fallback when Zarr is unavailable
```

The manifest records the complete configuration, Git revision, package versions,
grid metadata, execution mode, and an output inventory. The region table keeps
the model label as an explicit column and includes selection status plus summary
statistics from the raster fields.

Load one run without reading its large rasters:

```python
from analysis import load_run

run = load_run("runs/<run-directory>")
selected = run["regions"].query("selected")
print(run["summary"])
```

Open raster output lazily and plot the principal region distributions:

```python
from analysis import load_run, plot_run

run = load_run("runs/<run-directory>", load_rasters=True)
plot_run(run, selected_only=True, output_path="selected_distributions.png")
```

Combine all selected regions from an HPC parameter ensemble:

```python
from analysis import load_region_ensemble

regions = load_region_ensemble("runs", selected_only=True)
regions.groupby(["cohesion_eff", "soil_distribution"])["area"].describe()
```

The command-line analysis wrapper creates a combined CSV and one plot per run:

```bash
python analyse_landslide_outputs.py --runs runs --output analysis_output
```

For an interactive workflow, open
[`ShallowLandslider_output_analysis.ipynb`](ShallowLandslider_output_analysis.ipynb).
The notebook discovers run directories, builds run and region summaries,
compares parameter groups, saves distribution plots, and loads large raster
outputs only when requested. It writes tables as both CSV and indented JSON and
separately reports:

- selected initiation nodes;
- excavated source nodes;
- deposition endpoints;
- all runout-affected nodes;
- runout-only nodes outside the selected footprint;
- selected/runout overlap and the combined affected footprint;
- terminated path counts, branch-proportion checks, mass balance, and final
  soil-depth checks.

The intended small-model workflow is:

1. Run `ShallowLandslider_quickstart.ipynb` from top to bottom.
2. Leave `RUNS_ROOT = Path("runs")` and `RUN_INDEX = -1` in the analysis
   notebook to select the newest run.
3. Run `ShallowLandslider_output_analysis.ipynb` from top to bottom.
4. Inspect the readable JSON summaries, CSV tables, and figures in
   `analysis_output/`.

As an end-to-end check, the current 80 × 80 quick-start configuration produced
490 threshold-qualified sources, 487 moving/excavated sources, and 5,616
terminated Quinn branches. All per-source branch sums were one within floating-
point tolerance, erosion equalled deposition exactly, and no final soil depth
was negative. These counts are a reproducibility snapshot for the bundled DEM
and current example parameters, not fixed model constants.

See [CHANGELOG.md](CHANGELOG.md) for scientific-output implications and a
reverse-chronological description of every commit in the repository.



## **Configuration**
You can control simulation parameters via:
- **Component arguments** (e.g., `selection_method`, `random_seed`, `verbose`, etc.).
- **YAML config files** for reproducible CLI runs (see `ShallowLandslider_config.yaml`).

### **Main Parameters**
| Parameter                | Description                                      |
|-------------------------|--------------------------------------------------|
| `cohesion_eff`         | Soil effective cohesion (Pa)                     |
| `angle_int_frict`      | Internal friction angle (degrees)               |
| `submerged_soil_proportion` | Fraction of soil saturated                   |
| `pga_h`, `pga_v`       | Horizontal & vertical PGA arrays                |
| `selection_method`     | `"probabilistic"` or `"pga_weighted"`           |
| `proportion_method`    | `"empirical"`, `"conservative"`, etc.           |
| `random_seed`          | For reproducibility                             |
| `compute_displacement` | Calculate Newmark displacement for selected nodes |
| `displacement_threshold` | Minimum displacement for a node to become a runout source |
| `enable_runout`        | Enable the optional multiflow runout subcomponent |
| `update_soil`          | Allow runout to update source and endpoint soil depth |
| `verbose`              | Print progress and info (default: False)        |

---

## **Outputs**
After `run_one_step()`, the component populates:
- `landslide__factor_of_safety`
- `landslide__critical_acceleration`
- `landslide__driving_minus_critical_acceleration`
- `landslide__unstable_mask`
- `landslide__region_labels`
- `landslide__aspect_subgroup_labels`
- `landslide__dimension_split_labels` when KDE-informed splitting is enabled
- `landslide__selected_labels`
- `landslide__newmark_displacement` when displacement is enabled
- `landslide__erosion` when runout is enabled
- `landslide__deposition` when runout is enabled
- `landslide__soil_depth_change` when runout is enabled
- updated `soil__depth` when runout is enabled
- `ls.results["group_properties"]` (DataFrame with region metrics)
- `ls.results["runout"]` (multiflow sources, paths, proportions, and mass-change diagnostics)

---

## **Repository Structure**
```
ShallowLandslider/
├─ __init__.py
├─ components/
│  └─ shallow_landslider/
│     ├─ __init__.py
│     ├─ shallow_landslide_component.py # Main component
│     └─ shallow_landslide_runout.py    # Optional runout subcomponent
├─ utils/                               
│   ├─ __init__.py
│   ├─ utilities.py                     # Optional helpers (get_topo, generate_acceleration_grid, etc.)
├─ input_data/                          # Example DEMs and measured landslide data
│  ├─ dem/                              # Bundled example DEMs
│  ├─ nepal/, nz/, png/, japan/         # Example measured-landslide CSVs
├─ tests/                               # Pytest suite
├─ analysis/                            # Lazy run loading and ensemble analysis
├─ run_landslide_model_cli.py           # YAML-driven command line runner
├─ analyse_landslide_outputs.py         # Command-line output analysis
├─ ShallowLandslider_config.yaml        # Example CLI configuration
├─ ShallowLandslider_quickstart.ipynb   # Tutorial notebook
├─ ShallowLandslider_output_analysis.ipynb # Interactive output analysis
├─ CHANGELOG.md                         # Release, compatibility, and complete commit history
└─ README.md
```

---

## **Contributing**
1. Fork the repo and create a feature branch.
2. Follow PEP8 and use `ruff` for linting.
3. Add tests for new functionality (`pytest`).
4. Submit a pull request with a clear description.

---

## **License**
GNU GPLv3. See [LICENSE](LICENSE) for details.
