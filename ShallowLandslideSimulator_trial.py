# %% Load packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from shallow_landslide_component import ShallowLandslider

from utils import (
    get_topo,
    apply_soil_depth,
    pickle_or_not_to_pickle,
    calculate_terrain_attribute,
    generate_acceleration_grid,
    plot_comparison_panels_with_ecdf
)

from landlab.components import PriorityFloodFlowRouter
from landlab import imshowhs_grid  # to plot results

# %% Get measured data

model_region = "west"  # "west", "east", "south"

file_name_dict = {
    # Roback + Jones landslides - length/width
    "file1": "C:/Users/sghoshal/Documents/ArcGIS/Projects/landslides_Nepal/measuredLandslides_all.csv",
    # All landslides
    "file2": "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_spatialStats.csv",
}
region_configs = {
    "south": {
        "file3": "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_south_spatialStats.csv",
        "pickle_path": "measured_data_south.pkl",
        "modelled_data_folder": "pickled_runs_south",
    },
    "west": {
        "file3": "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_ZonalStats_clipbuffer.csv",
        "pickle_path": "measured_data.pkl",
        "modelled_data_folder": "pickled_runs",
    },
    "east": {
        "file3": "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_east_spatialStats.csv",
        "pickle_path": "measured_data_east.pkl",
        "modelled_data_folder": "pickled_runs_east",
    },
}

config = region_configs[model_region]
file_name_dict["file3"] = config["file3"]
pickle_path = config["pickle_path"]

measured_bundle = pickle_or_not_to_pickle(
    file_name_dict=file_name_dict, pickle_path=pickle_path
)
# measured_bundle = af.pickle_or_not_to_pickle(
#     file_name_dict=file_name_dict, pickle_path="measured_data_south.pkl"
# )
# measured_bundle = af.pickle_or_not_to_pickle(
#     file_name_dict=file_name_dict, pickle_path="measured_data.pkl"
# )

measured_data = measured_bundle["measured_data"]
measured_spatial_stats = measured_bundle["measured_spatial_stats"]
measured_spatial_stats_900greater = measured_bundle["measured_spatial_stats_900greater"]
measured_spatial_stats_clipped = measured_bundle["measured_spatial_stats_clipped"]
measured_spatial_stats_clipped_filt = measured_spatial_stats_clipped[measured_spatial_stats_clipped["Area_m2"]>900]

# Load measured length-width KDE for sampling
kde_dict = {
    "kde_data": measured_bundle["kde_data"],
    "kde_transform": measured_bundle["kde_transform"],
}
# %%
config_dict = {
    "dem_info": {
        "dem_type": "SRTMGL1",
        "buffer": 0.01,
        "smooth_num": 4,
        "plot_dem": True,
        "api_key": "f08b2664772eb044626d5cb114924de1",
    },
    "flow_params": {
        "flow_metric": "D8",
        "separate_hill_flow": True,
        "depression_handling": "fill",
        "update_hill_depressions": True,
        "accumulate_flow": True,
    },
    "soil_params": {
        "angle_int_frict": 30,  # degrees
        "cohesion_eff": 15e3,  # Pa
        "submerged_soil_proportion": 0.5,
        "max_soil_depth": 1.5,  # m
        "plot_soil": True,
        "distribution": "curvature",  # 'uniform', 'elevation', 'curvature', 'drainage_area', 'mean_elev_curv'
        # for "distribution" == "elevation", "relationship" == "linear", "exponential", "power", "sigmoid"
        # for "distribution" == "curvature", "relationship" == "linear", "linear_std_local", "linear_std_global", "piecewise"
        "relationship": "linear_std_local",  # only relevant for 'elevation' or 'curvature'
        "decay_rate": 1.0,
        "exponent": 2.0,
        # drainage_area-based params
        "drainage_transform": "threshold",
        "drainage_threshold": 1e6,
        "drainage_power": 0.3,
        # curvature-based params
        "P0": 0.05,
        "h_star": 1.0,
        "D": 0.01,
        "h_min": 0.1,
        "h_no_ss": 0.0,
    },
    "pga": {
        "horizontal_max": 0.6,
        "vertical_max": 0.2,
        "distribution": "uniform",
        "plot_grids": False,
    },
    "simulation": {
        "time_shaking": 10,  # seconds
        "displacement_threshold": 0,
        "aspect_interval": 20,
        "random_seed": 5000,  # keep if you want seed in file naming
        "handle_small_regions": "merge",
        "selection_method": "probabilistic",  # or 'pga_weighted'
        "proportion_method": "conservative",  # 'empirical', 'statistical', 'risk_profile', 'adaptive', or 'conservative'.
    },
    "plot_intermediates": {
        "factor_of_safety": False,
        "critical_acceleration": False,
        "unstable_areas": False,
        "filled_and_split": False,
    },
    "output": {
        "verbose": True,
        "save_plots": False,
        # "output_dir": "./pickled_runs_east/",  # defaults to current directory
        "save_pickle": True,
        "load_pickle": False,
    },
}

match model_region:
    case "west":
        config_dict["dem_info"]["north"] = 28.29
        config_dict["dem_info"]["east"] = 85.20
        config_dict["dem_info"]["south"] = 28.19
        config_dict["dem_info"]["west"] = 85.04
        config_dict["output"]["output_dir"] = "./pickled_runs/"
        config_dict["simulation"]["custom_proportion"] = None
    case "south":
        config_dict["dem_info"]["north"] = 27.55
        config_dict["dem_info"]["east"] = 85.36
        config_dict["dem_info"]["south"] = 27.43
        config_dict["dem_info"]["west"] = 85.20
        config_dict["output"]["output_dir"] = "./pickled_runs_south/"
        config_dict["simulation"]["custom_proportion"] = None
    case "east":
        config_dict["dem_info"]["north"] = 27.94
        config_dict["dem_info"]["east"] = 85.98
        config_dict["dem_info"]["south"] = 27.82
        config_dict["dem_info"]["west"] = 85.82
        config_dict["output"]["output_dir"] = "./pickled_runs_east/"
        config_dict["simulation"]["custom_proportion"] = 0.25

config_dict["output"]["output_dir"] = config["modelled_data_folder"]

# %% Build Landlab grid mg, set elevation/soil depth ...
load_dem = 'C:/Users/sghoshal/Documents/Python/nepal_cropped_dem.txt'

mg, z = get_topo(
    dem_type=config_dict["dem_info"]["dem_type"],
    north=config_dict["dem_info"]["north"],
    south=config_dict["dem_info"]["south"],
    east=config_dict["dem_info"]["east"],
    west=config_dict["dem_info"]["west"],
    buffer=config_dict["dem_info"]["buffer"],
    api_key=config_dict["dem_info"]["api_key"],
    smooth_num=config_dict["dem_info"]["smooth_num"],
    # load_dem=load_dem
)

# Initialize and run flow router
pf = PriorityFloodFlowRouter(
    mg,
    flow_metric=config_dict["flow_params"]["flow_metric"],
    separate_hill_flow=config_dict["flow_params"]["separate_hill_flow"],
    depression_handler=config_dict["flow_params"]["depression_handling"],
    update_hill_depressions=config_dict["flow_params"]["update_hill_depressions"],
    accumulate_flow=config_dict["flow_params"]["accumulate_flow"],
)
pf.run_one_step()

curv = calculate_terrain_attribute(
    grid=mg,
    field_name="topographic__elevation",
    attrib="planform_curvature",
)

# Add soil depth field if it doesn't exist
if "soil__depth" not in mg.at_node:
    soil_depth = mg.add_zeros("soil__depth", at="node")
    soil_depth = apply_soil_depth(
        mg,
        max_soil_depth=config_dict["soil_params"]["max_soil_depth"],
        distribution=config_dict["soil_params"]["distribution"],
        relationship=config_dict["soil_params"]["relationship"],
        # kwargs for various elevation-based soil depths
        decay_rate=config_dict["soil_params"]["decay_rate"],
        exponent=config_dict["soil_params"]["exponent"],
        # kwargs for various drainage area-based soil depths
        drainage_transform=config_dict["soil_params"]["drainage_transform"],
        drainage_threshold=config_dict["soil_params"]["drainage_threshold"],
        drainage_power=config_dict["soil_params"]["drainage_power"],
        # kwargs for curvature-based soil depth
        P0=config_dict["soil_params"]["P0"],
        h_star=config_dict["soil_params"]["h_star"],
        D=config_dict["soil_params"]["D"],
        h_min=config_dict["soil_params"]["h_min"],
        h_no_ss=config_dict["soil_params"]["h_no_ss"],
        plot=config_dict["soil_params"]["plot_soil"],
    )

# Add bedrock elevation field if it doesn't exist
if "bedrock__elevation" not in mg.at_node:
    mg.add_zeros("bedrock__elevation", at="node", clobber=True)

    # Set bedrock elevation
    mg.at_node["bedrock__elevation"][:] = (
        mg.at_node["topographic__elevation"] - mg.at_node["soil__depth"]
    )

# Calculate slopes
slopes = mg.calc_slope_at_node(elevs="topographic__elevation")
slopes_degrees = np.degrees(slopes)

# Calculate topographic aspect
aspect_nodes = np.array(
    mg.calc_aspect_at_node(
        elevs="topographic__elevation", unit="degrees", ignore_closed_nodes=True
    )
)

aspect_nodes[mg.boundary_nodes] = np.nan
aspect_nodes_array = aspect_nodes.reshape(mg.shape)

# %% Generate earthquake
pga_h, pga_v = generate_acceleration_grid(
    grid=mg,
    horizontal_max=config_dict["pga"]["horizontal_max"],
    vertical_max=config_dict["pga"]["vertical_max"],
    distribution=config_dict["pga"]["distribution"],
    plot_grids=config_dict["pga"]["plot_grids"],
)

# %% Load simulator

ls = ShallowLandslider(
    mg,
    cohesion_eff=config_dict["soil_params"]["cohesion_eff"],
    angle_int_frict=config_dict["soil_params"]["angle_int_frict"],
    submerged_soil_proportion=config_dict["soil_params"]["submerged_soil_proportion"],
    pga_h=pga_h,
    pga_v=pga_v,
    pga_h_max=config_dict["pga"]["horizontal_max"],
    pga_v_max=config_dict["pga"]["vertical_max"],
    selection_method=config_dict["simulation"]["selection_method"],
    proportion_method=config_dict["simulation"]["proportion_method"],
    random_seed=5000,
    custom_proportion=None,
    handle_small="merge",
    split_by_width_config={
        "kde_data": kde_dict["kde_data"],
        "kde_transform": kde_dict["kde_transform"],
        "convergence_threshold": 0.90,
        "min_region_size": 10,
        "max_iterations": 10,
        "width_threshold": 1.5,
    },
    g=9.81,
    verbose=config_dict["output"]["verbose"],
)
# %%%
if config_dict["soil_params"]["distribution"] == "curvature":
    plt.figure(layout="constrained", figsize=(12, 8))
    plt.subplot(121)
    imshowhs_grid(
        ls.grid,
        "topographic__elevation",
        plot_type="Drape1",
        drape1=np.ma.masked_invalid(ls.grid.at_node["planform_curvature"]),
        cmap="jet",
        allow_colorbar=True,
        cbar_or="vertical",
        ticks_km=True,
        cbar_loc="lower right",
        cbar_height=1.0,
        cbar_width=0.3,
    )

    plt.subplot(122)
    imshowhs_grid(
        ls.grid,
        "topographic__elevation",
        plot_type="Drape1",
        drape1=np.ma.masked_invalid(ls.grid.at_node["soil__depth"]),
        cmap="jet",
        allow_colorbar=True,
        cbar_or="vertical",
        ticks_km=True,
        cbar_loc="lower right",
        cbar_height=1.0,
        cbar_width=0.3,
    )
    # plt.suptitle('planform_curvature')

elif config_dict["soil_params"]["distribution"] == "drainage_area":
    plt.figure(layout="constrained", figsize=(12, 8))
    plt.subplot(121)
    imshowhs_grid(
        ls.grid,
        "topographic__elevation",
        plot_type="Drape1",
        drape1=np.ma.masked_invalid(
            np.ma.masked_greater(
                ls.grid.at_node["drainage_area"],
                config_dict["soil_params"]["drainage_threshold"],
            )
        ),
        cmap="jet",
        allow_colorbar=True,
        cbar_or="vertical",
        ticks_km=True,
        cbar_loc="lower right",
        cbar_height=1.0,
        cbar_width=0.3,
    )
    plt.subplot(122)
    imshowhs_grid(
        ls.grid,
        "topographic__elevation",
        plot_type="Drape1",
        drape1=np.ma.masked_invalid(ls.grid.at_node["soil__depth"]),
        cmap="jet",
        allow_colorbar=True,
        cbar_or="vertical",
        ticks_km=True,
        cbar_loc="lower right",
        cbar_height=1.0,
        cbar_width=0.3,
    )
    plt.suptitle("Drainage area")

elif config_dict["soil_params"]["distribution"] == "mean_elev_curv":
    plt.figure(layout="constrained", figsize=(12, 8))
    plt.subplot(221)
    imshowhs_grid(
        ls.grid,
        "topographic__elevation",
        plot_type="DEM",
        allow_colorbar=True,
        cbar_or="vertical",
        ticks_km=True,
        cbar_loc="lower right",
        cbar_height=1.0,
        cbar_width=0.3,
    )
    
    plt.subplot(222)
    imshowhs_grid(
        ls.grid,
        "topographic__elevation",
        plot_type="Drape1",
        drape1=np.ma.masked_invalid(ls.grid.at_node["planform_curvature"]),
        cmap="jet",
        allow_colorbar=True,
        cbar_or="vertical",
        ticks_km=True,
        cbar_loc="lower right",
        cbar_height=1.0,
        cbar_width=0.3,
    )

    plt.subplot(223)
    imshowhs_grid(
        ls.grid,
        "topographic__elevation",
        plot_type="Drape1",
        drape1=np.ma.masked_invalid(ls.grid.at_node["soil__depth"]),
        cmap="jet",
        allow_colorbar=True,
        cbar_or="vertical",
        ticks_km=True,
        cbar_loc="lower right",
        cbar_height=1.0,
        cbar_width=0.3,
    )
    plt.suptitle("Soil depth = mean of elevation-based and curvature-based")

plt.show()

# %% Error probes

# mask = ls.grid.at_node["landslide__unstable_mask"]
# print("Unstable nodes:", int(np.sum(mask)))
# print("Fraction unstable:", float(np.mean(mask)))
# print("a_driving min/max:", np.nanmin(ls._a_driving), np.nanmax(ls._a_driving))
# print("a_crit min/max:", np.nanmin(ls._a_transient), np.nanmax(ls._a_transient))

# print("H-PGA core mean:", float(np.nanmean(ls.grid.at_node["earthquake__horizontal_pga"][ls.grid.core_nodes])))
# print("V-PGA core mean:", float(np.nanmean(ls.grid.at_node["earthquake__vertical_pga"][ls.grid.core_nodes])))
# print("Any NaN at core H/V:",
#       np.isnan(ls.grid.at_node["earthquake__horizontal_pga"][ls.grid.core_nodes]).any(),
#       np.isnan(ls.grid.at_node["earthquake__vertical_pga"][ls.grid.core_nodes]).any())

# print("Cohesion_eff:", ls.cohesion_eff, "Pa")
# print("Friction angle (deg):", np.degrees(ls.angle_int_frict))
# slope_deg = np.degrees(ls.grid.calc_slope_at_node(elevs="topographic__elevation"))
# print("Slope deg min/median/max:", slope_deg.min(), np.median(slope_deg), slope_deg.max())

# print("'soil__depth' field exists:", "soil__depth" in ls.grid.at_node)
# if "soil__depth" in ls.grid.at_node:
#     h = ls.grid.at_node["soil__depth"]
#     print("Soil depth min/median/max:", float(np.nanmin(h)), float(np.nanmedian(h)), float(np.nanmax(h)))

# z = ls.grid.at_node["topographic__elevation"]
# print("Elevation NaNs present:", np.isnan(z).any())

# %%
ls.run_one_step()

# Access the group properties table
props = ls.results["group_properties"]  # pandas.DataFrame
print(props.head())

# Optional: export to CSV
# ls.export_group_properties("group_properties.csv")
# %%
# Get labels and reshape to 2-D for draping
labels = mg.at_node['landslide__selected_labels'].copy().reshape(mg.shape)

# Mask out zeros
labels_masked = np.ma.masked_where(labels == 0, labels)

# Make a copy of the cmap and set masked values to transparent
cmap = plt.cm.get_cmap('jet').copy()
cmap.set_bad(alpha=0)  # masked pixels won't be visible

props_filtered = props.loc[props["selected"]]

# %% Plot results

plot_comparison_panels_with_ecdf(
    observed_df=measured_spatial_stats_clipped_filt,
    model_df=props_filtered,
    mg=mg,                     # optional
    labels_masked=labels_masked,  # optional
    title=f"Modeled output: {config_dict['soil_params']['distribution']} - {config_dict['soil_params']['relationship']}",
    save_path=None             # or "comparison_panels.pdf"
)


# %%
count, bins_Roback = np.histogram(
    np.log10(measured_spatial_stats_clipped["Area_m2"]), 20
)

fig, axes = plt.subplots(2, 2, figsize=(18, 12), layout="constrained")

# Panel 1:
plt.sca(axes[0, 0])
imshowhs_grid(
    mg, 'topographic__elevation', plot_type='Drape1',
    drape1=labels_masked, cmap=cmap,
    allow_colorbar=True, cbar_or='vertical', ticks_km=True,
    cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3,
)

# Panel 2:
sns.histplot(
    data=props_filtered,
    x="area",
    label=f"Model - Selected areas ({len(props_filtered)})",
    legend=True,
    ax=axes[0, 1],
    bins=bins_Roback,
    log_scale=True,
    stat="density",
)

sns.histplot(
    data=measured_spatial_stats_clipped,
    x="Area_m2",
    label=f"Roback et al.; ({len(measured_spatial_stats_clipped)})",
    legend=True,
    ax=axes[0, 1],
    log_scale=True,
    bins=bins_Roback,
    stat="density",
)

axes[0, 1].set_xlabel('log10(Area) [m²]')
axes[0, 1].set_title('Histogram of Area')
axes[0, 1].legend()

# Panel 3:
sns.histplot(data=props_filtered, x='median_elevation', label="Model",
             ax=axes[1,0], stat='density', color='blue', alpha=0.6)
sns.histplot(data=measured_spatial_stats_clipped, x='mean_elev', label="Observed", ax=axes[1,0], stat='density', color='orange', alpha=0.6)
axes[1,0].set_xlabel('Elevation [m]')
axes[1,0].set_title('Histogram of Elevation')
axes[1,0].legend()

# Panel 4: Slope
sns.histplot(data=props_filtered, x='median_slope', label="Model",
             ax=axes[1,1], stat='density', color='blue', alpha=0.6)
sns.histplot(data=measured_spatial_stats_clipped, x='mean_slope', label="Observed", ax=axes[1,1], stat='density', color='orange', alpha=0.6)

axes[1,1].set_xlabel('Slope [degrees]')
axes[1,1].set_title('Histogram of Slope')
axes[1,1].legend()

fig.suptitle(f"Modeled output: {config_dict['soil_params']['distribution']} - {config_dict['soil_params']['relationship']}")

# %%
