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
import pandas as pd
import geopandas as gpd

import scipy.stats as stats
from scipy.stats import (
    wasserstein_distance
    )

from landlab import imshowhs_grid  # to plot results

from auxiliary_functions import (
    fit_bivariate_kde, calculate_excess_topography
    )

from inverse_gamma_script import compare_inverse_gamma

# %% Import measured data
# Import area, length and width data for all measured landslides in region
file_name = "C:/Users/sghoshal/Documents/ArcGIS/Projects/landslides_Nepal/measuredLandslides_all.csv"
measured_data = pd.read_csv(file_name)

# Import zonal statistics for Roback et al. 2017 landslides
file_name2 = "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_spatialStats.csv"
measured_spatial_stats = pd.read_csv(file_name2)

# Remove all landslides below 1000 m^2
measured_spatial_stats_900greater = measured_spatial_stats.drop(measured_spatial_stats[measured_spatial_stats['Area']<1000].index)

plot_order = ["Roback2017_Gorkha", "Jones2021_ASM"]

# Import Roback et al. 2017 landslide shapefile for test area
LSshapefile_name = 'C:/Users/sghoshal/Documents/ArcGIS/Projects/landslides_Nepal/landslide_Nepal_Roback.shp'
LSshapefile_file = gpd.read_file(LSshapefile_name)

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
# %%% Fitting bivariate data
kde_data, kde_transform = fit_bivariate_kde(dataframe=measured_data, x_col="length_m", 
                                y_col="width_m", category_col=None)

# %% ### Initialise and run ShallowLandslider

# %%% Initialise landslider
config_dict = {'dem_info': {
                    'dem_type': "SRTMGL1",
                    'north': 28.29, #31.34,# 28.29,
                    'east': 85.20, # 85.00, #103.70,
                    'south': 28.18, #31.23, # 28.18,
                    'west': 85.04, # 84.84, #103.56,
                    'buffer': 0.01,
                    'smooth_num': 4,
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
                    'distribution': 'elevation', # 'uniform' or 'elevation'
                    'plot_soil': False,
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
                    'split_convergence': 0.75, # threshold for splitting iterations
                    'min_region_size': 10, # minimum size of region to split
                    'selection_method': 'probabilistic', # or 'pga_weighted'
                    'proportion_method': 'statistical', # 'empirical', 'statistical', 'risk_profile', or 'adaptive'
                    },
                'plot_intermediates':{
                    'factor_of_safety': False,
                    'critical_acceleration': False,
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

# Load measured KDE for sampling
kde_dict = {
    'kde_data': kde_data,
    'kde_transform': kde_transform
    }
# %%% Run component

grid, model_results, model_grids = sim.run_one_step(kde_input=kde_dict)

# %%% Calculate excess topography
excess = calculate_excess_topography(grid=grid, method='morphological', )

# %%%% plot excess topography
plt.figure(layout='constrained')
imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1',
            drape1=np.ma.masked_invalid(excess),
            cmap='jet', allow_colorbar=True, cbar_or='vertical', ticks_km=True,
            cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3)
# plt.suptitle(f'Predicted landslides - {len(split_groups_props)}')
plt.show()

# %% ### Plot results ###

# Group properties after aspect splitting
subgroup_props = model_results['aspect_filtering']['subgroup_props']

# Group properties after width-splitting
split_groups_props = model_results['aspect_filtering']['dim_split_props']

# Groups after selection
selected_group_props = model_results['selected_landslides']['group_props']

# Displaced zones
displacement_zones = model_grids['transport_zones']
displacement_zone_props = model_results['sediment_transport']['transport_zone_props']

# %% Post-run length-width plot
plt.figure(layout='constrained')

ax_meas_scatter = sns.kdeplot(data=measured_data, x='length_m', y='width_m', color='red',
                        log_scale=(True, True), label='Measured landslide dimensions')
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

# %%% Map of predicted landslides
plt.figure(layout='constrained')
imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1',
            drape1=np.ma.masked_invalid(np.ma.masked_equal(model_results['aspect_filtering']['dim_split_groups'], 0)),
            cmap='jet', allow_colorbar=True, cbar_or='vertical', ticks_km=True,
            cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3)
plt.suptitle(f'Predicted landslides - {len(split_groups_props)}')
plt.show()

# %%

plt.figure(layout='constrained')
imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1',
            drape1=np.ma.masked_invalid(np.ma.masked_equal(model_grids['selected_landslides'], 0)),
            cmap='jet', allow_colorbar=True, cbar_or='vertical', ticks_km=True,
            cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3)
plt.suptitle(f'Predicted landslides - {len(selected_group_props)}')
plt.show()
# %%
plt.figure(layout='constrained')
imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1',
            drape1=np.ma.masked_invalid(np.ma.masked_equal(model_grids['transport_zones'], 0)),
            cmap='jet', allow_colorbar=True, cbar_or='vertical', ticks_km=True,
            cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3)
plt.suptitle('Predicted landslides - post-displacement')
plt.show()



# %%%% Map of soil depth change
plt.figure(layout='constrained')
imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1', ticks_km=True,
            drape1=np.ma.masked_equal(model_grids['soil_depth_change'], 0.0),
            allow_colorbar=False, cmap='viridis', altdeg=45, azdeg=315)
plt.suptitle('Change in soil depth')

# %%%%  Plot magnitude-frequency for selected landslides
RobackData_greaterthan900 = LSshapefile_file["SHAPE_Area"][LSshapefile_file["SHAPE_Area"]>900]
count, bins_Roback = np.histogram(np.log10(RobackData_greaterthan900), 20)

fig_mag_freq, ax_mag_freq = plt.subplots(layout='constrained')
sns.histplot(data=subgroup_props, x="area", label="Model - All areas",
            legend=True, ax=ax_mag_freq, bins=bins_Roback, log_scale=True, stat='density')
# sns.histplot(data=split_groups_props, x="area", label="Model - All split areas",
#             legend=True, ax=ax_mag_freq, bins=bins_Roback, log_scale=True, stat='density')
sns.histplot(data=selected_group_props, x="area", label="Model - Selected areas",
            legend=True, ax=ax_mag_freq, bins=bins_Roback, log_scale=True, stat='density')
# sns.histplot(data=displacement_zone_props, x="area", label="Model - displaced areas",
#             legend=True, ax=ax_mag_freq, bins=bins_Roback, log_scale=True, stat='density')
sns.histplot(x=RobackData_greaterthan900, label="Roback et al. (>900 $m^2$)",
            legend=True, ax=ax_mag_freq, log_scale=True, bins=bins_Roback, stat='density')

# ax_mag_freq.set_xscale("log")
ax_mag_freq.legend()
ax_mag_freq.set_xlabel("Area")

# %%%%% Plot KDE for magnitude-frequency
fig_mag_freq_2, ax_mag_freq_2 = plt.subplots(layout='constrained')
sns.kdeplot(data=subgroup_props, x="area", label=f"Model - All areas ({len(subgroup_props)})",
            legend=True, ax=ax_mag_freq_2, log_scale=True, color='grey')
sns.kdeplot(data=split_groups_props, x="area", label=f"Model - All split areas ({len(split_groups_props)})",
            legend=True, ax=ax_mag_freq_2, log_scale=True, color='grey')
sns.kdeplot(data=selected_group_props, x="area", label=f"Model - Selected areas ({len(selected_group_props)})",
            legend=True, ax=ax_mag_freq_2, log_scale=True, color='grey')
sns.kdeplot(data=displacement_zone_props, x="area", label=f"Model - Displaced areas ({len(displacement_zone_props)})",
            legend=True, ax=ax_mag_freq_2, log_scale=True, color='red')
sns.kdeplot(x=RobackData_greaterthan900, label="Roback et al. (>900 $m^2$)",
            legend=True, ax=ax_mag_freq_2, log_scale=True, color='grey')

# ax_mag_freq.set_xscale("log")
ax_mag_freq_2.legend()
ax_mag_freq_2.set_xlabel("Area")
# %%%% Plot other parameter distributions
# Elevation
plt.figure(layout='constrained')
sns.histplot(x=grid.at_node['topographic__elevation'], color='grey', alpha=0.25, stat='density', label='Regional elevations')
sns.histplot(data=selected_group_props, x='median_elevation', label="Model elevations (mean)", stat='density')
sns.histplot(data=measured_spatial_stats_900greater, x='Elevation_mean', label='Measured landslides', stat='density')

plt.legend()
plt.title("Landslides vs. elevation")
plt.xlabel("Elevation (m)")

# Slope
plt.figure(layout='constrained')
sns.histplot(x=model_grids['slopes'], color='grey', alpha=0.25, stat='density', label='Regional slopes')
sns.histplot(data=selected_group_props, x='median_slope', label="Median slopes of unstable areas", stat='density',
            log_scale=False)
sns.histplot(data=measured_spatial_stats_900greater, x='Slope_deg_mean', label='Measured landslides', stat='density')

plt.legend()
plt.title("Landslides vs. Slope")
plt.xlabel("Slope ($\degree$)")
# %% Compare variables
def compare_continuous_variables(observed_df, modeled_df, column_mapping):
    """
    Compare continuous variables with different column names between observed and modeled data.
    `column_mapping` should be a dict: {observed_col: modeled_col}
    """
    def check_scale_bias(obs_data, mod_data, var_name):
        """Check for systematic scale differences"""
        print(f"\n🔍 Scale Bias Analysis for {var_name}:")
        obs_range = obs_data.max() - obs_data.min()
        mod_range = mod_data.max() - mod_data.min()
        range_ratio = mod_range / obs_range if obs_range != 0 else np.inf
        
        print(f"Range ratio (modeled/observed): {range_ratio:.3f}")
        print(f"Observed range: {obs_range:.3f}, Modeled range: {mod_range:.3f}")
        
        # Check quantile differences
        print("Quantile Comparison:")
        for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
            obs_q = obs_data.quantile(q)
            mod_q = mod_data.quantile(q)
            diff = mod_q - obs_q
            print(f"  Q{q:.1f} - Observed: {obs_q:.3f}, Modeled: {mod_q:.3f}, Diff: {diff:+.3f}")
        
        return {'range_ratio': range_ratio, 'obs_range': obs_range, 'mod_range': mod_range}

    def landslide_specific_analysis(obs_data, mod_data, var_name):
        """Landslide-focused additional analysis"""
        print(f"\nLandslide-Specific Analysis for {var_name}:")
        
        # Extreme value analysis (important for landslide susceptibility)
        obs_95 = obs_data.quantile(0.95)
        mod_95 = mod_data.quantile(0.95)
        obs_99 = obs_data.quantile(0.99)
        mod_99 = mod_data.quantile(0.99)
        
        print("Extreme values comparison:")
        print(f"  95th percentile - Observed: {obs_95:.3f}, Modeled: {mod_95:.3f}, Diff: {mod_95-obs_95:+.3f}")
        print(f"  99th percentile - Observed: {obs_99:.3f}, Modeled: {mod_99:.3f}, Diff: {mod_99-obs_99:+.3f}")
        
        # Variable-specific analysis
        if 'slope' in var_name.lower():
            print("Slope-specific analysis:")
            for threshold in [15, 30, 45]:  # Common slope thresholds for landslides
                obs_steep_pct = (obs_data > threshold).mean() * 100
                mod_steep_pct = (mod_data > threshold).mean() * 100
                diff_pct = mod_steep_pct - obs_steep_pct
                print(f"  Slopes >{threshold}° - Observed: {obs_steep_pct:.1f}%, Modeled: {mod_steep_pct:.1f}%, Diff: {diff_pct:+.1f}%")
        
        elif 'elevation' in var_name.lower():
            print("Elevation-specific analysis:")
            # High elevation analysis (often more landslide-prone)
            obs_mean = obs_data.mean()
            high_elev_threshold = obs_mean + obs_data.std()
            obs_high_pct = (obs_data > high_elev_threshold).mean() * 100
            mod_high_pct = (mod_data > high_elev_threshold).mean() * 100
            diff_pct = mod_high_pct - obs_high_pct
            print(f"  High elevation (>{high_elev_threshold:.0f}m) - Observed: {obs_high_pct:.1f}%, Modeled: {mod_high_pct:.1f}%, Diff: {diff_pct:+.1f}%")
        
        return {
            'percentile_95': {'observed': obs_95, 'modeled': mod_95},
            'percentile_99': {'observed': obs_99, 'modeled': mod_99}
        }

    results = {}

    print("=== CONTINUOUS VARIABLES COMPARISON ===")
    print("Focusing on robust tests for landslide data (often non-normal distributions)\n")

    for obs_col, mod_col in column_mapping.items():
        if obs_col not in observed_df.columns or mod_col not in modeled_df.columns:
            print(f"Warning: '{obs_col}' or '{mod_col}' not found in one or both datasets")
            continue

        print(f"\n--- Analysis for {obs_col} vs {mod_col} ---")

        obs_data = observed_df[obs_col].dropna()
        mod_data = modeled_df[mod_col].dropna()

        results[obs_col] = {}

        # Descriptive Stats
        print(f"Observed - Mean: {obs_data.mean():.3f}, Median: {obs_data.median():.3f}, Std: {obs_data.std():.3f}, N: {len(obs_data)}")
        print(f"Modeled  - Mean: {mod_data.mean():.3f}, Median: {mod_data.median():.3f}, Std: {mod_data.std():.3f}, N: {len(mod_data)}")

        # Scale bias analysis
        scale_results = check_scale_bias(obs_data, mod_data, obs_col)
        results[obs_col]['scale_bias'] = scale_results

        # Landslide-specific analysis
        landslide_results = landslide_specific_analysis(obs_data, mod_data, obs_col)
        results[obs_col]['landslide_specific'] = landslide_results

        # Normality check
        if len(obs_data) < 5000 and len(mod_data) < 5000:
            obs_normal_p = stats.shapiro(obs_data)[1]
            mod_normal_p = stats.shapiro(mod_data)[1]
            print(f"\nNormality check: Observed p={obs_normal_p:.4f}, Modeled p={mod_normal_p:.4f}")
            normal_assumption = obs_normal_p > 0.05 and mod_normal_p > 0.05
        else:
            print("\nLarge sample size - skipping normality test")
            normal_assumption = False

        print("\nStatistical Tests:")
        # Mann-Whitney U
        mw_stat, mw_pval = stats.mannwhitneyu(obs_data, mod_data, alternative='two-sided')
        print(f"- Mann-Whitney U test: U={mw_stat:.0f}, p={mw_pval:.4f}")
        results[obs_col]['mann_whitney'] = {'statistic': mw_stat, 'p_value': mw_pval, 'recommended': True}

        # Kolmogorov-Smirnov
        ks_stat, ks_pval = stats.ks_2samp(obs_data, mod_data)
        print(f"- Kolmogorov-Smirnov test: D={ks_stat:.4f}, p={ks_pval:.4f}")
        results[obs_col]['ks_test'] = {'statistic': ks_stat, 'p_value': ks_pval, 'recommended': True}

        # Welch's t-test if normal
        if normal_assumption:
            welch_stat, welch_pval = stats.ttest_ind(obs_data, mod_data, equal_var=False)
            print(f"- Welch's t-test: t={welch_stat:.4f}, p={welch_pval:.4f}")
            results[obs_col]['welch_test'] = {'statistic': welch_stat, 'p_value': welch_pval}
        else:
            print("Welch's t-test skipped (non-normal)")

        # Correlation
        min_len = min(len(obs_data), len(mod_data))
        obs_sample = obs_data.sample(n=min_len, random_state=42)
        mod_sample = mod_data.sample(n=min_len, random_state=42)

        spearman_r, spearman_p = stats.spearmanr(obs_sample, mod_sample)
        print(f"- Spearman correlation: rho={spearman_r:.4f}, p={spearman_p:.4f}")
        results[obs_col]['spearman'] = {'correlation': spearman_r, 'p_value': spearman_p}

        if normal_assumption:
            pearson_r, pearson_p = stats.pearsonr(obs_sample, mod_sample)
            print(f"- Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.4f}")
            results[obs_col]['pearson'] = {'correlation': pearson_r, 'p_value': pearson_p}

        print("=" * 80)

    return results

def create_comparison_plots(observed_df, modeled_df, column_mapping, log_scale=None):
    """
    Create visual comparison plots between observed and modeled data for each variable.
    `column_mapping`: dict of observed_col -> modeled_col
    `log_scale`: list of booleans for each variable, or None for auto-detection
    """
    
    def ecdf(data):
        x = np.sort(data)
        y = np.arange(1, len(x)+1) / len(x)
        return x, y
    
    n_cols = len(column_mapping)
    
    # Set up log_scale if not provided
    if log_scale is None:
        log_scale = [False] * n_cols
    elif len(log_scale) != n_cols:
        log_scale = log_scale + [False] * (n_cols - len(log_scale))
    
    # Create 4-row plot grid (added scatter plot)
    fig, axes = plt.subplots(4, n_cols, figsize=(5 * n_cols, 12),
                            layout='constrained')

    if n_cols == 1:
        axes = axes.reshape(4, 1)

    for i, (obs_col, mod_col) in enumerate(column_mapping.items()):
        obs_data = observed_df[obs_col].dropna()
        mod_data = modeled_df[mod_col].dropna()

        # Row 1: Histogram
        axes[0, i].hist(obs_data, alpha=0.7, label='Observed', bins=30, density=True, color='blue')
        axes[0, i].hist(mod_data, alpha=0.7, label='Modeled', bins=30, density=True, color='red')
        axes[0, i].set_title(f'{obs_col} vs {mod_col}\nHistogram')
        axes[0, i].legend()
        if log_scale[i]:
            axes[0, i].set_xscale('log')

        # Row 2: Q-Q Plot against Weibull distribution
        min_len = min(len(obs_data), len(mod_data))
        obs_sample = np.random.choice(obs_data, min_len, replace=False)
        mod_sample = np.random.choice(mod_data, min_len, replace=False)

        # Fit Weibull distribution to observed data to get reference parameters
        try:
            # Fit Weibull distribution (scipy uses Weibull_min which is standard 2-parameter Weibull)
            weibull_params = stats.weibull_min.fit(obs_sample, floc=0)
            shape_param, loc_param, scale_param = weibull_params
            
            # Generate theoretical quantiles for both datasets
            prob_points = np.linspace(0.01, 0.99, len(obs_sample))
            weibull_quantiles = stats.weibull_min.ppf(prob_points, shape_param, loc=loc_param, scale=scale_param)
            
            # Sort both samples for Q-Q plot
            obs_sorted = np.sort(obs_sample)
            mod_sorted = np.sort(mod_sample)
            
            # Plot Q-Q against Weibull
            axes[1, i].scatter(weibull_quantiles, obs_sorted, alpha=0.7, color='blue', label='Observed', s=15)
            axes[1, i].scatter(weibull_quantiles, mod_sorted, alpha=0.7, color='red', label='Modeled', s=15)
            
            # Add reference line
            axes[1, i].plot(weibull_quantiles, weibull_quantiles, 'k--', alpha=0.8, label='Perfect fit')
            
            axes[1, i].set_title(f'{obs_col} vs {mod_col}\nQ-Q Plot vs Weibull\n(shape={shape_param:.2f}, scale={scale_param:.2f})')
            axes[1, i].set_xlabel('Weibull Theoretical Quantiles')
            axes[1, i].set_ylabel('Sample Quantiles')
            
        except Exception:
            # Fallback to empirical Q-Q plot if Weibull fitting fails
            print(f"Warning: Weibull fitting failed for {obs_col}, using empirical Q-Q plot")
            obs_sorted = np.sort(obs_sample)
            mod_sorted = np.sort(mod_sample)
            
            axes[1, i].scatter(obs_sorted, mod_sorted, alpha=0.7, color='purple', s=15)
            min_val = min(np.min(obs_sorted), np.min(mod_sorted))
            max_val = max(np.max(obs_sorted), np.max(mod_sorted))
            axes[1, i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='1:1 line')
            axes[1, i].set_title(f'{obs_col} vs {mod_col}\nEmpirical Q-Q Plot')
            axes[1, i].set_xlabel('Observed Quantiles')
            axes[1, i].set_ylabel('Modeled Quantiles')
        
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        
        # Row 3: Empirical CDF plot
        x_obs, y_obs = ecdf(obs_data)
        x_mod, y_mod = ecdf(mod_data)
        axes[2, i].step(x_obs, y_obs, label='Observed', where='post', color='blue')
        axes[2, i].step(x_mod, y_mod, label='Modeled', where='post', color='red')
        wassert_dist = wasserstein_distance(obs_data, mod_data)
        axes[2, i].set_title(f'{obs_col} vs {mod_col}\nEmpirical CDF\nWasserstein Distance: {wassert_dist:.4f}')
        axes[2, i].set_xlabel(obs_col)
        axes[2, i].set_ylabel('CDF')
        axes[2, i].legend()
        if log_scale[i]:
            axes[2, i].set_xscale('log')

        # Row 4: Percentile-Percentile Scatter Plot
        percentiles = np.arange(5, 100, 5)
        obs_percentiles = [obs_data.quantile(p/100) for p in percentiles]
        mod_percentiles = [mod_data.quantile(p/100) for p in percentiles]
        
        axes[3, i].scatter(obs_percentiles, mod_percentiles, alpha=0.7, s=30, color='purple')
        
        # Add 1:1 line
        min_val = min(min(obs_percentiles), min(mod_percentiles))
        max_val = max(max(obs_percentiles), max(mod_percentiles))
        axes[3, i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line')
        
        # Calculate R² for percentiles
        r_squared = np.corrcoef(obs_percentiles, mod_percentiles)[0, 1]**2
        
        axes[3, i].set_xlabel(f'Observed {obs_col} Percentiles')
        axes[3, i].set_ylabel(f'Modeled {mod_col} Percentiles')
        axes[3, i].set_title(f'{obs_col} vs {mod_col}\nPercentile-Percentile Plot\nR² = {r_squared:.3f}')
        axes[3, i].legend()
        axes[3, i].grid(True, alpha=0.3)

    # Add overall title if config_dict exists, otherwise use generic title
    try:
        plt.suptitle(f"Landslide Data Comparison - {config_dict['soil_params']['distribution']}", fontsize=14)
    except (NameError, KeyError):
        plt.suptitle("Landslide Data Comparison", fontsize=14)
    
    plt.show()

# %%%%
column_mapping = {
    # observed_col: modeled_col
    'Elevation_mean':'median_elevation',
    'Slope_deg_mean':'median_slope'
}

# Continuous:
comparison_results = compare_continuous_variables(measured_spatial_stats_900greater,
                            selected_group_props, column_mapping)
# %%%%
# Plots:
create_comparison_plots(measured_spatial_stats_900greater, selected_group_props, column_mapping, 
                        log_scale=[True, False, False])
# %%
import numpy as np
import matplotlib.pyplot as plt
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
compare_area_distributions(measured_spatial_stats_900greater['Area'],
                           selected_group_props['area'], distributions=["invgamma"])

# %%
compare_inverse_gamma(data1=measured_spatial_stats_900greater['Area'],
                      data2=selected_group_props['area'])
# %%
