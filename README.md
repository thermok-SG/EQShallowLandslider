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
- **Optional runout routing** that redistributes failed soil along hill-flow receiver paths after Newmark displacement is computed

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
Runout is available as an optional soil-depth update after selected landslides have Newmark displacement values. To enable it through `ShallowLandslider`, the grid must first be routed with `PriorityFloodFlowRouter` using `separate_hill_flow=True`, which creates the required `hill_flow__receiver_node` and `hill_flow__receiver_proportions` fields.

Runout is only executed when all three component flags are enabled:
- `compute_displacement=True`
- `enable_runout=True`
- `update_soil=True`

In the YAML CLI config, this means using `chunking.mode: never`, keeping
`flow_params.enable` and `flow_params.separate_hill_flow` true, and setting
`simulation.compute_displacement`, `simulation.enable_runout`, and
`simulation.update_soil` to true. Runout modifies `soil__depth` in place and
caches diagnostic erosion/deposition arrays on the runout subcomponent.

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
```

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

See [CHANGELOG.md](CHANGELOG.md) for scientific-output implications and the Git
commit associated with the large-DEM curvature fixes.



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
- updated `soil__depth` when runout is enabled
- `ls.results["group_properties"]` (DataFrame with region metrics)

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
├─ run_landslide_model_cli.py           # YAML-driven command line runner
├─ ShallowLandslider_config.yaml        # Example CLI configuration
├─ ShallowLandslider_quickstart.ipynb   # Tutorial notebook
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
