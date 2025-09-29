"""
Modeled data analysis


"""

# %% Load required packages
import auxiliary_functions as af
import data_analysis as da

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import scipy.stats as stats

from landlab import imshowhs_grid  # to plot results

# %% Load measured data
file_name_dict = {
    # Roback + Jones landslides - length/width
    "file1": "C:/Users/sghoshal/Documents/ArcGIS/Projects/landslides_Nepal/measuredLandslides_all.csv",
    # All landslides
    "file2": "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_spatialStats.csv",
    # Clipped landslides
    "file3": "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_ZonalStats_clipbuffer.csv",
    # "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_ZonalStats_clipbuffer.csv",
    # "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_south_spatialStats.csv",
    "shapefile_name": "C:/Users/sghoshal/Documents/ArcGIS/Projects/landslides_Nepal/landslide_Nepal_Roback.shp",
}
measured_bundle = af.pickle_or_not_to_pickle(
    file_name_dict=file_name_dict, pickle_path="measured_data.pkl"
)

measured_data = measured_bundle["measured_data"]
measured_spatial_stats = measured_bundle["measured_spatial_stats"]
measured_spatial_stats_900greater = measured_bundle["measured_spatial_stats_900greater"]
measured_spatial_stats_clipped = measured_bundle["measured_spatial_stats_clipped"]

# Load measured length-width KDE for sampling
kde_dict = {
    "kde_data": measured_bundle["kde_data"],
    "kde_transform": measured_bundle["kde_transform"],
}

measured_spatial_stats_clipped.drop(
    measured_spatial_stats_clipped[
        measured_spatial_stats_clipped["Area_m2"] < 900
    ].index,
    inplace=True,
)
count, bins_Roback = np.histogram(
    np.log10(measured_spatial_stats_clipped["Area_m2"]), 20
)

# %% Load modelled data
modelled_data_folder = "pickled_runs"

modelled_runs = af.load_all_runs(modelled_data_folder)
print(f"Loaded data: {modelled_runs.keys()}")

# %%% Prepare data for analysis

model_dfs_dict = da.extract_selected_group_props(modelled_runs, name_style="string")

column_mapping = {
    # observed_col: modeled_col
    "Area_m2": "area",
    "mean_elev": "median_elevation",
    "mean_slope": "median_slope",
}

# %% Compare modelled data with measured
da.compare_all_models(
    observed_df=measured_spatial_stats_clipped,
    model_dfs_dict=model_dfs_dict,
    column_mapping=column_mapping,
    skip_missing_columns=True,
)

# %%
da.create_all_models_comparison(
    observed_df=measured_spatial_stats_clipped,
    model_dfs_dict=model_dfs_dict,
    column_mapping=column_mapping,
)

# %% Plot heatmap

all_metrics = da.create_performance_summary(
    observed_df=measured_spatial_stats_clipped,
    model_dfs_dict=model_dfs_dict,
    column_mapping=column_mapping,
)
# %%
da.plot_histograms_ecdfs_combined(
    observed_df=measured_spatial_stats_clipped,
    model_dfs_dict=model_dfs_dict,
    column_mapping=column_mapping,
    custom_names=True,
)
# %%
