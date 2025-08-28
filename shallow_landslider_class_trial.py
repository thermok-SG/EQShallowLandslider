# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:21:12 2025

@author: sghoshal
"""
# %%
# Load class and components
from shallow_landslider_class import ShallowLandslideSimulator

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import scipy.stats as stats

from landlab import imshowhs_grid  # to plot results

from auxiliary_functions import (
    pickle_or_not_to_pickle
    )

from inverse_gamma_script import compare_inverse_gamma

# %% Get measured data
bundle = pickle_or_not_to_pickle("measured_data.pkl")

measured_data = bundle["measured_data"]
measured_spatial_stats_900greater = bundle["measured_spatial_stats_900greater"]
measured_spatial_stats_clipped = bundle["measured_spatial_stats_clipped"]

# Load measured length-width KDE for sampling
kde_dict = {
    'kde_data': bundle["kde_data"],
    'kde_transform': bundle["kde_transform"]
    }

# %%% Length vs. width of measured data

plt.figure(layout='constrained')
ax_meas_scatter = sns.scatterplot(data=measured_data, x='length_m', y='width_m', hue='name')

plt.axline([0,0],[1,1], label='1:1')

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Landslide length (m)')
plt.ylabel('Landslide width (m)')


# %%% KDE: Measured data
plt.figure(layout='constrained')
ax_bivar = sns.kdeplot(data=measured_data, x='length_m', y='width_m', color='gray',
                        log_scale=(True, True), label='Measured landslide dimensions')
# sns.kdeplot(data=measured_data, x='length_m', y='width_m', hue='name',
#                        legend=True, ax=ax_bivar)

plt.axline([1,1],[10,10], label='1:1', linestyle='--', color='black')

plt.xlabel('Landslide length (m)')
plt.ylabel('Landslide width (m)')

# %% ### Initialise and run ShallowLandslider

# %%% Initialise landslider
config_dict = {
    'dem_info': {
        'dem_type': "SRTMGL1",
        'north': 28.29, #31.34,# 28.29,
        'east': 85.20, # 85.00, #103.70,
        'south': 28.18, #31.23, # 28.18,
        'west': 85.04, # 84.84, #103.56,
        'buffer': 0.01,
        'plot_dem' : True
        },
    'flow_params': {
        'flow_metric': 'D8',
        'separate_hill_flow': True,
        'depression_handling': 'fill',
        'update_hill_depressions': True,
        'accumulate_flow': True
        },
    'soil_params': {
        'angle_int_frict': np.radians(30),
        'cohesion_eff': 15e3,  # Pa
        'submerged_soil_proportion': 0.5,
        'max_soil_depth': 1.5, # m
        'distribution': 'drainage_area', # 'uniform', 'elevation' or 'drainage_area'
        'relationship': 'linear', # 'linear', 'exponential', 'power', 'sigmoid'
        'decay_rate': 1.0, # rate of decay of exponential function
        'exponent': 2.0, # exponent for when relationship == 'power'
        'drainage_transform': 'threshold', # transformation for drainage area values, can be 'log', 'sqrt', 'power', 'threshold'
        'drainage_threshold': None, # drainage area threshold when 'drainage_transform'=='threshold'
        'drainage_power': 0.3, # power value for when 'drainage_transform'=='power'
        'plot_soil': True,
        },
    'pga': {
        'horizontal_max': 0.6,
        'vertical_max': 0.2,
        'distribution': "uniform",
        'plot_grids': False
        },
    'simulation': {
        'time_shaking': 10,  # seconds
        'displacement_threshold': 0,
        'aspect_interval': 20,
        'random_seed': 5000, # for reproducibility
        'handle_small_regions': 'merge', # what happens to 1px regions: 'keep', 'merge', or 'remove'
        'split_convergence': 0.75, # threshold for splitting iterations
        'min_region_size': 10, # minimum size of region to split
        'selection_method': 'probabilistic', # or 'pga_weighted'
        'proportion_method': 'statistical', # 'empirical', 'statistical', 'risk_profile', or 'adaptive'
        },
    'plot_intermediates':{
        'factor_of_safety': True,
        'critical_acceleration': True,
        'unstable_areas': False, # Issue here
        'filled_and_split': True
    },
    'output': {
        'save_plots': False,
        'output_dir': None,
        }
    }

# Initialise component with given parameters
sim = ShallowLandslideSimulator(config=config_dict)

# Loads DEM for the class
sim.load_dem()

# %%% Plot drainage area
if config_dict['soil_params']['distribution'] == 'drainage_area':
    
    plt.figure(layout='constrained')
    imshowhs_grid(sim.grid, "topographic__elevation", plot_type='Drape1',
                drape1=np.ma.masked_invalid(np.ma.masked_greater(sim.grid.at_node['drainage_area'], 10e5)),
                cmap='jet', allow_colorbar=True, cbar_or='vertical', ticks_km=True,
                cbar_loc='lower right', cbar_height=1.0, cbar_width=0.3)
    plt.suptitle('Drainage area')
    
    plt.show()

# %%% Plot regional parameter distributions
# Elevation
plt.figure(layout='constrained')
sns.histplot(x=sim.grid.at_node['topographic__elevation'], color='grey', alpha=0.25, stat='density', label='Regional elevations')
sns.histplot(data=measured_spatial_stats_900greater, x='mean_elev', label='All measured landslides', stat='density')
sns.histplot(data=measured_spatial_stats_clipped, x='mean_elev', label='Clipped measured landslides', stat='density')

plt.legend()
plt.title("Elevation distribution")
plt.xlabel("Elevation (m)")

# Slope
plt.figure(layout='constrained')
sns.histplot(x=sim.slopes_degrees, color='grey', alpha=0.25, stat='density', label='Regional slopes')
sns.histplot(data=measured_spatial_stats_900greater, x='mean_slope', label='All measured landslides', stat='density')
sns.histplot(data=measured_spatial_stats_clipped, x='mean_slope', label='Clipped measured landslides', stat='density')


plt.legend()
plt.title("Slope distribution")
plt.xlabel("Slope ($\degree$)")

# %%% Run component

grid, model_results, model_grids = sim.run_one_step(kde_input=kde_dict)

# %% ### Plot results ###

# Group properties after aspect splitting
subgroup_props = model_results['aspect_filtering']['subgroup_props']

# Group properties after width-splitting
split_groups_props = model_results['aspect_filtering']['dim_split_props']

# Groups after selection
selected_group_props = model_results['selected_landslides']['group_props']

# # Displaced zones
# displacement_zones = model_grids['transport_zones']
# displacement_zone_props = model_results['sediment_transport']['transport_zone_props']

# %% Post-run length-width plot
fig_meas_scatter, ax_meas_scatter = plt.subplots(layout='constrained')

# Add KDE background first (so it's behind the points)
# kde_plotter.add_kde_background(ax_meas_scatter, category=None, levels=15, alpha=0.3, colors='gray')

sns.kdeplot(data=measured_data, x='length_m', y='width_m', color='red',
                        log_scale=(True, True), label='Measured landslide dimensions', ax=ax_meas_scatter)
# sns.scatterplot(data=subgroup_props, x='slope_direction_length', y='perpendicular_width')
sns.scatterplot(data=subgroup_props, x='slope_direction_length_new', y='perpendicular_width_new',
                label='Pre-split groups', ax=ax_meas_scatter)
sns.scatterplot(data=split_groups_props, x='slope_direction_length_new', y='perpendicular_width_new',
                label="Split groups", ax=ax_meas_scatter)
sns.scatterplot(data=selected_group_props, x='slope_direction_length_new', y='perpendicular_width_new',
                label=f"Selected groups - {len(selected_group_props)}", ax=ax_meas_scatter)
# sns.scatterplot(data=displacement_zone_props, x='slope_direction_length_new', y='perpendicular_width_new',
#                 label=f"Displacement groups - {len(displacement_zone_props)}", ax=ax_meas_scatter)

plt.axline([0,0],[1,1], label='1:1')

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel('Landslide length (m)')
plt.ylabel('Landslide width (m)')

# %%% Maps of predicted landslides
# Post-displacement landslides
plt.figure(layout='constrained')
imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1',
            drape1=np.ma.masked_invalid(np.ma.masked_equal(model_results['aspect_filtering']['dim_split_groups'], 0)),
            cmap='jet', allow_colorbar=True, cbar_or='vertical', ticks_km=True,
            cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3)
plt.suptitle(f'Predicted landslides - {len(split_groups_props)}')

# Selected landslides
plt.figure(layout='constrained')
imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1',
            drape1=np.ma.masked_invalid(np.ma.masked_equal(model_grids['selected_landslides'], 0)),
            cmap='binary', allow_colorbar=True, cbar_or='vertical', ticks_km=True,
            cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3)
plt.suptitle(f'Predicted & selected landslides - {len(selected_group_props)}')

plt.show()

# # %%
# plt.figure(layout='constrained')
# imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1',
#             drape1=np.ma.masked_invalid(np.ma.masked_equal(model_grids['transport_zones'], 0)),
#             cmap='jet', allow_colorbar=True, cbar_or='vertical', ticks_km=True,
#             cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3)
# plt.suptitle('Predicted landslides - post-displacement')
# plt.show()

# %%%% Map of soil depth change
plt.figure(layout='constrained')
imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1', ticks_km=True,
            drape1=np.ma.masked_equal(sim.grid.at_node['soil__depth'], 0.0),
            allow_colorbar=True, cmap='viridis', altdeg=45, azdeg=315, cbar_or='vertical',
            cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3)
plt.suptitle('Soil depth')

# %%%%  Plot magnitude-frequency for selected landslides
# RobackData_greaterthan900 = measured_spatial_stats_clipped['Area_m2'][measured_spatial_stats_clipped['Area_m2']>900]
measured_spatial_stats_clipped.drop(measured_spatial_stats_clipped[measured_spatial_stats_clipped['Area_m2']<900].index, inplace=True)
# LSshapefile_file["SHAPE_Area"][LSshapefile_file["SHAPE_Area"]>900]
# count, bins_Roback = np.histogram(np.log10(RobackData_greaterthan900), 20)
count, bins_Roback = np.histogram(np.log10(measured_spatial_stats_clipped['Area_m2']), 20)
# %%%%
fig_mag_freq, ax_mag_freq = plt.subplots(layout='constrained')
sns.histplot(data=subgroup_props, x="area", label=f"Model - All areas ({len(subgroup_props)})",
            legend=True, ax=ax_mag_freq, bins=bins_Roback, log_scale=True, stat='density')
# sns.histplot(data=split_groups_props, x="area", label="Model - All split areas",
#             legend=True, ax=ax_mag_freq, bins=bins_Roback, log_scale=True, stat='density')
sns.histplot(data=selected_group_props, x="area", label=f"Model - Selected areas ({len(selected_group_props)})",
            legend=True, ax=ax_mag_freq, bins=bins_Roback, log_scale=True, stat='density')
# sns.histplot(data=displacement_zone_props, x="area", label="Model - displaced areas",
#             legend=True, ax=ax_mag_freq, bins=bins_Roback, log_scale=True, stat='density')
sns.histplot(data=measured_spatial_stats_clipped, x='Area_m2', label=f"Roback et al.; ({len(measured_spatial_stats_clipped)})",
            legend=True, ax=ax_mag_freq, log_scale=True, bins=bins_Roback, stat='density')

# ax_mag_freq.set_xscale("log")
ax_mag_freq.legend()
ax_mag_freq.set_xlabel("Area")

# %%%%% Plot KDE for magnitude-frequency
fig_mag_freq_2, ax_mag_freq_2 = plt.subplots(layout='constrained')
sns.kdeplot(data=subgroup_props, x="area", label=f"Model - All areas ({len(subgroup_props)})",
            legend=True, ax=ax_mag_freq_2, log_scale=True)
sns.kdeplot(data=split_groups_props, x="area", label=f"Model - All split areas ({len(split_groups_props)})",
            legend=True, ax=ax_mag_freq_2, log_scale=True)
sns.kdeplot(data=selected_group_props, x="area", label=f"Model - Selected areas ({len(selected_group_props)})",
            legend=True, ax=ax_mag_freq_2, log_scale=True)
# sns.kdeplot(data=displacement_zone_props, x="area", label=f"Model - Displaced areas ({len(displacement_zone_props)})",
#             legend=True, ax=ax_mag_freq_2, log_scale=True, color='red')
sns.kdeplot(data=measured_spatial_stats_clipped, x='Area_m2', label="Roback et al. (>900 $m^2$)",
            legend=True, ax=ax_mag_freq_2, log_scale=True)

# ax_mag_freq.set_xscale("log")
ax_mag_freq_2.legend()
ax_mag_freq_2.set_xlabel("Area")
# %%%% Plot other parameter distributions
# Elevation
count, elevation_bins = np.histogram(measured_spatial_stats_clipped['mean_elev'], 20)
plt.figure(layout='constrained')
sns.histplot(x=grid.at_node['topographic__elevation'], color='grey', alpha=0.25, stat='density', label='Regional elevations')
# sns.histplot(data=measured_spatial_stats_900greater, x='Elevation_mean', label='All measured landslides', stat='density')
sns.histplot(data=measured_spatial_stats_clipped, x='mean_elev', label='Clipped measured landslides', stat='density',
            bins=elevation_bins)
sns.histplot(data=selected_group_props, x='median_elevation', label="Model elevations (mean)", stat='density', bins=elevation_bins)

plt.legend()
plt.title("Landslides vs. elevation")
plt.xlabel("Elevation (m)")

# Slope
count, slope_bins = np.histogram(measured_spatial_stats_clipped['mean_slope'], 20)
plt.figure(layout='constrained')
sns.histplot(x=model_grids['slopes'], color='grey', alpha=0.25, stat='density', label='Regional slopes')
# sns.histplot(data=measured_spatial_stats_900greater, x='Slope_deg_mean', label='All measured landslides', stat='density')
sns.histplot(data=measured_spatial_stats_clipped, x='mean_slope',
            label='Clipped measured landslides', stat='density', bins=slope_bins)
sns.histplot(data=selected_group_props, x='median_slope',
            label="Median slopes of unstable areas", stat='density', bins=slope_bins)

plt.legend()
plt.title("Landslides vs. Slope")
plt.xlabel("Slope ($\degree$)")
# %% Compare variables
# --- Utility functions ---
def pick_reference_distribution(var_name):
    """Pick appropriate reference distribution for QQ plot."""
    lname = var_name.lower()
    if "area" in lname:
        return "lognorm"
    elif "slope" in lname:
        return "weibull_min"
    elif "elev" in lname:
        return "norm"
    else:
        return "empirical"

# --- Comparison function ---
def compare_continuous_variables(observed_df, modeled_df, column_mapping):
    """
    Compare continuous variables between observed and modeled data.
    Focused on landslide datasets (often skewed, heavy-tailed).
    """
    results = {}
    print("\n================ LANDSLIDE VARIABLE COMPARISON ================")
    
    for obs_col, mod_col in column_mapping.items():
        if obs_col not in observed_df.columns or mod_col not in modeled_df.columns:
            print(f"⚠️ Skipping {obs_col}: not found in one or both datasets")
            continue

        obs = observed_df[obs_col].dropna()
        mod = modeled_df[mod_col].dropna()

        print("\n------------------------------------------------------------")
        print(f"📊 Variable: {obs_col} (Observed) vs {mod_col} (Modeled)")
        print("------------------------------------------------------------")

        results[obs_col] = {}

        # --- Descriptive stats ---
        print("[Summary Stats]")
        print(f"Observed: mean={obs.mean():.2f}, median={obs.median():.2f}, std={obs.std():.2f}, n={len(obs)}")
        print(f"Modeled : mean={mod.mean():.2f}, median={mod.median():.2f}, std={mod.std():.2f}, n={len(mod)}")

        # --- Range & quantile comparison ---
        obs_range, mod_range = obs.max() - obs.min(), mod.max() - mod.min()
        range_ratio = mod_range / obs_range if obs_range != 0 else np.inf
        print("\n[Range & Quantiles]")
        print(f"Range ratio (Modeled/Observed): {range_ratio:.3f}")
        for q in [0.1, 0.5, 0.9]:
            o_q, m_q = obs.quantile(q), mod.quantile(q)
            print(f"Q{int(q*100)}: Obs={o_q:.2f}, Mod={m_q:.2f}, Δ={m_q-o_q:+.2f}")

        results[obs_col]['range_ratio'] = range_ratio

        # --- Tail analysis ---
        print("\n[Tail Analysis]")
        for p in [0.95, 0.99]:
            o_q, m_q = obs.quantile(p), mod.quantile(p)
            print(f"{int(p*100)}th pct: Obs={o_q:.2f}, Mod={m_q:.2f}, Δ={m_q-o_q:+.2f}")

        # --- Geology-specific checks ---
        if "slope" in obs_col.lower():
            print("\n[Slope thresholds]")
            for t in [15, 30, 45]:
                o_pct = (obs > t).mean()*100
                m_pct = (mod > t).mean()*100
                print(f">{t}°: Obs={o_pct:.1f}%, Mod={m_pct:.1f}%, Δ={m_pct-o_pct:+.1f}%")

        if "elev" in obs_col.lower():
            high_thresh = obs.mean() + obs.std()
            o_pct = (obs > high_thresh).mean()*100
            m_pct = (mod > high_thresh).mean()*100
            print(f"\n[Elevation > mean+σ ≈ {high_thresh:.0f}m]: Obs={o_pct:.1f}%, Mod={m_pct:.1f}%, Δ={m_pct-o_pct:+.1f}%")

        # --- Statistical tests ---
        print("\n[Statistical Tests]")
        mw_stat, mw_p = stats.mannwhitneyu(obs, mod, alternative='two-sided')
        ks_stat, ks_p = stats.ks_2samp(obs, mod)
        w_dist = stats.wasserstein_distance(obs, mod)

        print(f"Mann–Whitney U: U={mw_stat:.0f}, p={mw_p:.4g}")
        print(f"Kolmogorov–Smirnov: D={ks_stat:.3f}, p={ks_p:.4g}")
        print(f"Wasserstein distance: {w_dist:.3f}")

        results[obs_col]['mann_whitney'] = mw_p
        results[obs_col]['ks'] = ks_p
        results[obs_col]['wasserstein'] = w_dist

    print("\n================ END COMPARISON ================\n")
    return results


# --- Plotting function ---
def create_comparison_plots(observed_df, modeled_df, column_mapping):
    n_cols = len(column_mapping)
    fig, axes = plt.subplots(3, n_cols, figsize=(5*n_cols, 9), layout='constrained')
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    for i, (obs_col, mod_col) in enumerate(column_mapping.items()):
        obs = observed_df[obs_col].dropna()
        mod = modeled_df[mod_col].dropna()

        # 1. Histogram + KDE
        axes[0,i].hist(obs, bins=30, density=True, alpha=0.6, label='Observed', color='blue')
        axes[0,i].hist(mod, bins=30, density=True, alpha=0.6, label='Modeled', color='red')
        axes[0,i].set_title(f"{obs_col} vs {mod_col}\nHistogram")
        axes[0,i].legend()

        # 2. QQ Plot vs reference distribution
        dist_name = pick_reference_distribution(obs_col)
        try:
            if dist_name != 'empirical':
                params = getattr(stats, dist_name).fit(obs)
                ref_dist = getattr(stats, dist_name)
                quantiles = np.linspace(0.01,0.99,100)
                ref_q = ref_dist.ppf(quantiles,*params)
                obs_q = np.quantile(obs, quantiles)
                mod_q = np.quantile(mod, quantiles)
                axes[1,i].scatter(ref_q, obs_q, color='blue', alpha=0.6, label='Observed')
                axes[1,i].scatter(ref_q, mod_q, color='red', alpha=0.6, label='Modeled')
                axes[1,i].plot(ref_q, ref_q, 'k--', lw=1)
                axes[1,i].set_title(f"QQ Plot vs {dist_name}")
            else:
                obs_q = np.quantile(obs, np.linspace(0.01,0.99,100))
                mod_q = np.quantile(mod, np.linspace(0.01,0.99,100))
                axes[1,i].scatter(obs_q, mod_q, color='purple', alpha=0.6)
                axes[1,i].plot([min(obs_q), max(obs_q)], [min(obs_q), max(obs_q)], 'r--')
                axes[1,i].set_title("Empirical QQ Plot")
        except Exception:
            axes[1,i].text(0.5,0.5,"QQ failed", ha='center')

        # 3. ECDF comparison
        def ecdf(x):
            x = np.sort(x)
            y = np.arange(1, len(x)+1)/len(x)
            return x,y
        xo,yo = ecdf(obs)
        xm,ym = ecdf(mod)
        w_dist = stats.wasserstein_distance(obs, mod)
        axes[2,i].step(xo,yo, label='Observed', color='blue')
        axes[2,i].step(xm,ym, label='Modeled', color='red')
        axes[2,i].set_title(f"ECDF\nWasserstein={w_dist:.3f}")
        axes[2,i].legend()

        # 🔑 Apply log-scale if it's an area variable
        if "area" in obs_col.lower() or "area" in mod_col.lower():
            for row in range(3):
                axes[row,i].set_xscale('log')

    plt.suptitle("Landslide Data: Observed vs Modeled", fontsize=14)
    plt.show()



# %%%%
column_mapping = {
    # observed_col: modeled_col
    'mean_elev':'median_elevation',
    'mean_slope':'median_slope'
}
column_mapping = {
    # observed_col: modeled_col
    'Area_m2' : 'area',
    'mean_elev':'median_elevation',
    'mean_slope':'median_slope'
}

# Continuous:
comparison_results = compare_continuous_variables(measured_spatial_stats_clipped,
                            selected_group_props, column_mapping)
# %%%%
# Plots:
create_comparison_plots(measured_spatial_stats_clipped, selected_group_props, column_mapping)
#                        log_scale=[True, False, False])
# %%

from scipy.stats import ks_2samp
import warnings

def compute_gof(data, dist_name, params, n):
    dist = getattr(stats, dist_name)
    log_likelihood = np.sum(dist.logpdf(data, *params))
    k = len(params)
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return log_likelihood, aic, bic

def compare_area_distributions(data1, data2, labels=("Measured", "Modelled"), distributions=None):
    if distributions is None:
        distributions = ['invgamma', 'lognorm', 'gamma', 'weibull_min', 'pareto']
    
    datasets = [data1, data2]
    results = {}

    fig, axs = plt.subplots(1, 2, figsize=(18, 7), layout='constrained')
    
    for i, data in enumerate(datasets):
        data = np.asarray(data)
        n = len(data)
        x = np.logspace(np.log10(data.min()), np.log10(data.max()), 500)
        sorted_data = np.sort(data)
        label = labels[i]
        
        axs[i].hist(data, bins=np.logspace(np.log10(data.min()), np.log10(data.max()), 30), 
                    density=True, alpha=0.4, color='grey', label=f"{label} histogram")
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
        axs[i].set_title(f"{label} Distribution Fits")
        axs[i].set_xlabel("Area")
        axs[i].set_ylabel("PDF")

        results[label] = {}

        for dist_name in distributions:
            try:
                dist = getattr(stats, dist_name)
                if dist_name == 'invgamma':
                    params = dist.fit(data, floc=0)
                else:
                    params = dist.fit(data)
                
                pdf_vals = dist.pdf(x, *params)
                axs[i].plot(x, pdf_vals, label=f"{dist_name}", lw=2)
                
                logL, aic, bic = compute_gof(data, dist_name, params, n)
                results[label][dist_name] = {
                    "params": params,
                    "logL": logL,
                    "AIC": aic,
                    "BIC": bic
                }
            except Exception as e:
                warnings.warn(f"Could not fit {dist_name} to {label}: {e}")
    
        axs[i].legend()
    
    plt.show()

    # Print GoF table
    print("\nGoodness-of-Fit Metrics (Lower AIC/BIC = better fit):\n")
    for label in labels:
        print(f"\n--- {label} ---")
        print(f"{'Distribution':<15}{'LogL':>12}{'AIC':>12}{'BIC':>12}")
        for dist_name, metrics in results[label].items():
            print(f"{dist_name:<15}{metrics['logL']:>12.2f}{metrics['AIC']:>12.2f}{metrics['BIC']:>12.2f}")

    # KS test
    ks_stat, ks_p_value = ks_2samp(data1, data2)
    print("\n--- Kolmogorov-Smirnov Test Between Datasets ---")
    print(f"KS Statistic: {ks_stat:.4f}")
    print(f"p-value: {ks_p_value:.4f}")
    if ks_p_value > 0.05:
        print("Distributions are statistically similar (fail to reject H0).")
    else:
        print("Distributions are statistically different (reject H0).")


# %%%
compare_area_distributions(measured_spatial_stats_clipped['area'],
                        selected_group_props['area'], distributions=["invgamma"])

# %%
compare_inverse_gamma(data1=measured_spatial_stats_900greater['Area'],
                    data2=selected_group_props['area'])
# %%
