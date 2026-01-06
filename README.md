# **ShallowLandslider: A Landlab Component for Earthquake-Induced Landslide Simulation**

## **Overview**
`ShallowLandslider` is a _component_ designed to model the distribution of shallow landslides triggered by seismic shaking. It integrates physics-based stability calculations, probabilistic selection methods, and optional width-based recursive splitting to simulate landslide occurrence and distribution across a DEM.

This tool is intended for **geomorphologists, hazard modellers, and Earth scientists** who need reproducible, scalable landslide simulations. It's landlab-based architecture means that it can be easily integrated within larger landscape evolution modelling scenarios.

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

## **Installation**
### **Requirements**
- Python ≥ 3.9
- [Landlab](https://landlab.readthedocs.io/)
- NumPy, Pandas, SciPy, scikit-image

## **Measured Data & KDE Bundle (Recommended)**
Many workflows **benefit strongly** from measured landslide data to guide *statistically based region splitting* and to compare modelled vs observed characteristics. While technically optional, this step is **highly recommended** for realistic run outcomes.

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
- **CSV columns** for KDE fitting: `length_m`, `width_m`. Values must be **positive** if log-transform is enabled (default).
- If you enable grouping (`category_col`), ensure that each category has **≥ 5 samples**—sparser groups are skipped.
- The function `fit_bivariate_kde(...)` internally called by `pickle_or_not_to_pickle` uses `scipy.stats.gaussian_kde` and supports optional log-transforms and per-category bandwidths.

### Troubleshooting
- `ValueError: Cannot log-transform ... contains values <= 0` → turn off log-transform or filter/clean your data.
- If your CSV column names differ, pass the correct `x_col`/`y_col` into `fit_bivariate_kde(...)` when you build the pickle the first time.
- Use `verbose=True` to see a concise progress log while building/loading the pickle.

## **Quick start**
```python
import numpy as np
import matplotlib.pyplot as plt
from shallow_landslide_component import ShallowLandslider

from helper_functions import (
    get_topo,
    apply_soil_depth,
    pickle_or_not_to_pickle,
    calculate_terrain_attribute,
    generate_acceleration_grid,
)

from landlab.components import PriorityFloodFlowRouter
from landlab import imshowhs_grid  # to plot results

# 0. Load measured data
file_name_dict = {
    "file1": "/path/to/measuredLandslides_all.csv",     # area/length/width table for all measured landslides
    "file2": "/path/to/measuredLandslides_spatialStats.csv",    # zonal stats for all measured landslides
    "file3": "/path/to/region_spatialStats.csv",        # region-clipped zonal stats (optional but recommended if using subregion)
}

bundle = pickle_or_not_to_pickle(file_name_dict, pickle_path="measured_data_east.pkl", min_area=900, verbose=True)

# Access KDE for recursive split
kde_dict = {
    "kde_data": bundle["kde_data"],
    "kde_transform": bundle["kde_transform"],
}

# 1. Build a Landlab grid (example: fetch DEM from OpenTopography)
grid, z = get_topo(north=27.94, south=27.82, east=85.98, west=85.82, buffer=0.01, api_key="<API_KEY>")

# 2. Initialize and run flow router
pf = PriorityFloodFlowRouter(
    mg,
    flow_metric=config_dict["flow_params"]["flow_metric"],
    separate_hill_flow=config_dict["flow_params"]["separate_hill_flow"],
    depression_handler=config_dict["flow_params"]["depression_handling"],
    update_hill_depressions=config_dict["flow_params"]["update_hill_depressions"],
    accumulate_flow=config_dict["flow_params"]["accumulate_flow"],
)
pf.run_one_step()

# 3. Apply soil depth
curv = calculate_terrain_attribute(
    grid=mg,
    field_name="topographic__elevation",
    attrib="planform_curvature",
)
soil_depth = apply_soil_depth(grid, max_soil_depth=1.5, distribution="curvature")

# 4. Generate earthquake PGA grids
pga_h, pga_v = generate_acceleration_grid(grid, horizontal_max=0.6, vertical_max=0.2)

# 4. Instantiate and run the component
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
    # KDE-based splitting
    split_by_width_config={
        "kde_data": kde_dict["kde_data"],
        "kde_transform": kde_dict["kde_transform"],
        "split_convergence": 0.75, # stops recursive splitting once proportion of regions match expectations
        "min_region_size": 10, # minimum number of pixels a region must have to be a split candidate
        "max_iterations": 10, # max number of recursive iterations to prevent infinite loops
        "width_threshold": 1.5, # Ratio of actual width / expected width beyond which a region is flagged for splitting
    }
)

ls.run_one_step()

# 5. Access results
print(ls.results["group_properties"].head())
```



## **Configuration**
You can control simulation parameters via:
- **Component arguments** (e.g., `selection_method`, `random_seed`, `verbose`, etc. in configuration dictionary).
- **JSON config files** for reproducible runs (see `example_config.json` in repo).

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
- `landslide__selected_labels`
- `landslide__probability`
- `landslide__metadata`
- `group_properties` (DataFrame with region metrics)

---

## **Repository Structure**
```
ShallowLandslider/
├─ __init__.py
├─ shallow_landslide_component.py   # Main component
├─ helper_functions/
│   ├─ __init__.py
│   ├─ displacement.py              # Newmark displacement
│   ├─ regions.py                   # Region labelling & properties
│   ├─ selection.py                 # Probabilistic & PGA selection
│   ├─ split.py                     # Recursive splitting
│   ├─ stability.py                 # FoS & critical acceleration
│   ├─ utilities.py                 # Optional helpers (get_topo, generate_acceleration_grid, etc.)
├─ examples/                        # Suite of example data/notebooks to run full model
├─ tests/                           # Pytest suite
├─ example_config.json              # Sample config
└─ README.md                        # This file
```

---

## **Contributing**
1. Fork the repo and create a feature branch.
2. Follow PEP8 and use `ruff` for linting.
3. Add tests for new functionality (`pytest`).
4. Submit a pull request with a clear description.

---

## **License**
GNU GPLv3 .See [LICENSE](LICENSE) for details.