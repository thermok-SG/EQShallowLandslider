"""
Modeled data analysis


"""
# %% Load required packages
import auxiliary_functions as af
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import scipy.stats as stats

from landlab import imshowhs_grid  # to plot results

# %% Load measured data
file_name_dict = {
    # Roback + Jones landslides - length/width
    'file1': "C:/Users/sghoshal/Documents/ArcGIS/Projects/landslides_Nepal/measuredLandslides_all.csv",
    # All landslides
    'file2': "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_spatialStats.csv",
    # Clipped landslides
    'file3': "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_ZonalStats_clipbuffer.csv",
        # "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_ZonalStats_clipbuffer.csv",
        # "C:/Users/sghoshal/Documents/ArcGIS/Projects/Landslides_Nepal_Main/Roback2017_south_spatialStats.csv",
    
    'shapefile_name': "C:/Users/sghoshal/Documents/ArcGIS/Projects/landslides_Nepal/landslide_Nepal_Roback.shp"
}
measured_bundle = af.pickle_or_not_to_pickle(file_name_dict=file_name_dict, pickle_path="measured_data.pkl")

measured_data = measured_bundle["measured_data"]
measured_spatial_stats = measured_bundle["measured_spatial_stats"]
measured_spatial_stats_900greater = measured_bundle["measured_spatial_stats_900greater"]
measured_spatial_stats_clipped = measured_bundle["measured_spatial_stats_clipped"]

# Load measured length-width KDE for sampling
kde_dict = {
    'kde_data': measured_bundle["kde_data"],
    'kde_transform': measured_bundle["kde_transform"]
    }

measured_spatial_stats_clipped.drop(measured_spatial_stats_clipped[measured_spatial_stats_clipped['Area_m2']<900].index, inplace=True)
count, bins_Roback = np.histogram(np.log10(measured_spatial_stats_clipped['Area_m2']), 20)

# %% Load modelled data
modelled_data_folder =  "pickled_runs"

modelled_runs = af.load_all_runs(modelled_data_folder)
print(f"Loaded data: {modelled_runs.keys()}")

# Extract dataframes
selected_group_props_curv_lin = modelled_runs["SRTMGL1_c15000_curvature_linear_seed5000"]["selected_group_props"]
selected_group_props_curv_linstdglo = modelled_runs["SRTMGL1_c15000_curvature_linear_std_global_seed5000"]["selected_group_props"]
selected_group_props_curv_linstdloc = modelled_runs["SRTMGL1_c15000_curvature_linear_std_local_seed5000"]["selected_group_props"]
selected_group_props_drainage = modelled_runs["SRTMGL1_c15000_drainage_area_seed5000"]["selected_group_props"]
selected_group_props_elev_lin = modelled_runs["SRTMGL1_c15000_elevation_linear_seed5000"]["selected_group_props"]
selected_group_props_uniform = modelled_runs["SRTMGL1_c15000_uniform_seed5000"]["selected_group_props"]

# %%%
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
    
def compare_all_models(observed_df, model_dfs_dict, column_mapping):
    """Compare observed data against multiple model runs."""
    summary_results = {}
    
    for model_name, model_df in model_dfs_dict.items():
        print(f"Model: {model_name}")
        results = compare_continuous_variables(observed_df, model_df, column_mapping)
        summary_results[model_name] = results
    
    # Create ranking table
    metrics = ['wasserstein', 'ks', 'mann_whitney']
    for variable in column_mapping.keys():
        print(f"\n{variable} Rankings:")
        for metric in metrics:
            scores = [(name, results[variable][metric]) for name, results in summary_results.items()]
            scores.sort(key=lambda x: x[1])  # Lower is better for these metrics
            print(f"  {metric}: {[name for name, score in scores]}")
            
def create_all_models_comparison(observed_df, model_dfs_dict, column_mapping):
    """
    Create plots comparing observed data against all model runs.
    
    Parameters:
    -----------
    observed_df : pandas.DataFrame
        The observed/measured landslide data
    model_dfs_dict : dict
        Dictionary with model names as keys and dataframes as values
        e.g., {'Model1': df1, 'Model2': df2, ...}
    column_mapping : dict
        Mapping of observed columns to model columns
    """
    n_vars = len(column_mapping)
    n_models = len(model_dfs_dict)
    
    # Create figure with subplots: one row per variable, one column per model + observed
    fig, axes = plt.subplots(n_vars, n_models + 1, 
                           figsize=(4*(n_models + 1), 4*n_vars), 
                           layout='constrained')
    
    # Handle case of single variable
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    model_names = list(model_dfs_dict.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, n_models + 1))
    
    for var_idx, (obs_col, mod_col) in enumerate(column_mapping.items()):
        obs_data = observed_df[obs_col].dropna()
        
        # Plot observed data in first column
        axes[var_idx, 0].hist(obs_data, bins=30, density=True, alpha=0.7, 
                             color=colors[0], label='Observed')
        axes[var_idx, 0].set_title(f'Observed\n{obs_col}')
        axes[var_idx, 0].set_ylabel('Density')
        
        # Apply log scale for area variables
        if "area" in obs_col.lower():
            axes[var_idx, 0].set_xscale('log')
        
        # Plot each model in subsequent columns
        for model_idx, (model_name, model_df) in enumerate(model_dfs_dict.items()):
            col_idx = model_idx + 1
            mod_data = model_df[mod_col].dropna()
            
            # Histogram
            axes[var_idx, col_idx].hist(mod_data, bins=30, density=True, alpha=0.7,
                                      color=colors[col_idx], label=model_name)
            
            # Overlay observed for comparison (lighter)
            axes[var_idx, col_idx].hist(obs_data, bins=30, density=True, alpha=0.3,
                                      color='gray', label='Observed (ref)')
            
            # Calculate and display Wasserstein distance
            w_dist = stats.wasserstein_distance(obs_data, mod_data)
            axes[var_idx, col_idx].text(0.05, 0.95, f'W={w_dist:.2f}', 
                                      transform=axes[var_idx, col_idx].transAxes,
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            axes[var_idx, col_idx].set_title(f'{model_name}\n{mod_col}')
            axes[var_idx, col_idx].legend(fontsize=8)
            
            if "area" in obs_col.lower():
                axes[var_idx, col_idx].set_xscale('log')
        
        # Set common x-axis label for bottom row
        if var_idx == n_vars - 1:
            for col_idx in range(n_models + 1):
                axes[var_idx, col_idx].set_xlabel(obs_col)
    
    plt.suptitle('Distribution Comparison: Observed vs All Models', fontsize=16)
    plt.show()


def create_overlay_comparison(observed_df, model_dfs_dict, column_mapping):
    """
    Create overlay plots with all models on same axes.
    Better for detailed comparison but can get crowded.
    """
    n_vars = len(column_mapping)
    fig, axes = plt.subplots(2, n_vars, figsize=(5*n_vars, 10), layout='constrained')
    
    if n_vars == 1:
        axes = axes.reshape(2, 1)
    
    model_names = list(model_dfs_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names) + 1))
    
    for var_idx, (obs_col, mod_col) in enumerate(column_mapping.items()):
        obs_data = observed_df[obs_col].dropna()
        
        # Top row: Histograms
        axes[0, var_idx].hist(obs_data, bins=30, density=True, alpha=0.6, 
                            color=colors[0], label='Observed', linewidth=2)
        
        # Bottom row: ECDFs
        def ecdf(x):
            x_sorted = np.sort(x)
            y = np.arange(1, len(x_sorted)+1) / len(x_sorted)
            return x_sorted, y
        
        x_obs, y_obs = ecdf(obs_data)
        axes[1, var_idx].step(x_obs, y_obs, color=colors[0], linewidth=2, 
                            label='Observed', where='post')
        
        # Plot all models
        wasserstein_scores = []
        for model_idx, (model_name, model_df) in enumerate(model_dfs_dict.items()):
            mod_data = model_df[mod_col].dropna()
            color = colors[model_idx + 1]
            
            # Histogram
            axes[0, var_idx].hist(mod_data, bins=30, density=True, alpha=0.5,
                                color=color, label=model_name)
            
            # ECDF
            x_mod, y_mod = ecdf(mod_data)
            axes[1, var_idx].step(x_mod, y_mod, color=color, linewidth=1.5,
                                label=model_name, where='post')
            
            # Calculate Wasserstein distance
            w_dist = stats.wasserstein_distance(obs_data, mod_data)
            wasserstein_scores.append((model_name, w_dist))
        
        # Formatting
        if "area" in obs_col.lower():
            axes[0, var_idx].set_title(f'{obs_col} - Exceedance Probability')
            # Add power law reference line
            if len(obs_data) > 10:  # Only if we have enough data
                x_min, x_max = axes[0, var_idx].get_xlim()
                x_ref = np.logspace(np.log10(x_min), np.log10(x_max), 100)
                y_ref = (x_ref / np.median(obs_data)) ** (-1.4)
                y_ref = np.clip(y_ref, 1e-4, 1)  # Reasonable bounds
                axes[0, var_idx].loglog(x_ref, y_ref, '--', color='black', 
                                       alpha=0.4, linewidth=1, label='β=-1.4 ref')
        else:
            axes[0, var_idx].set_title(f'{obs_col} - Histograms')
        
        axes[0, var_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        axes[1, var_idx].set_title(f'{obs_col} - ECDFs')
        axes[1, var_idx].set_ylabel('Cumulative Probability')
        axes[1, var_idx].set_xlabel(obs_col)
        axes[1, var_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Apply log scale for area variables (bottom row too)
        if "area" in obs_col.lower():
            axes[1, var_idx].set_xscale('log')
        
        # Add Wasserstein scores as text
        score_text = '\n'.join([f'{name}: {score:.2f}' for name, score in wasserstein_scores])
        axes[1, var_idx].text(0.02, 0.98, f'Wasserstein distances:\n{score_text}', 
                            transform=axes[1, var_idx].transAxes,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=8)
    
    plt.suptitle('Distribution Overlay: All Models vs Observed', fontsize=16)
    plt.show()


def create_performance_summary(observed_df, model_dfs_dict, column_mapping):
    """
    Create a summary heatmap of performance metrics across all models.
    """
    import pandas as pd
    
    # Calculate metrics for all models
    metrics_data = []
    
    for model_name, model_df in model_dfs_dict.items():
        for obs_col, mod_col in column_mapping.items():
            obs_data = observed_df[obs_col].dropna()
            mod_data = model_df[mod_col].dropna()
            
            # Calculate multiple metrics
            ks_stat, ks_p = stats.ks_2samp(obs_data, mod_data)
            w_dist = stats.wasserstein_distance(obs_data, mod_data)
            
            # Normalized metrics (0-1 scale, lower is better)
            metrics_data.append({
                'Model': model_name,
                'Variable': obs_col,
                'KS_statistic': ks_stat,
                'Wasserstein': w_dist / (obs_data.std() + mod_data.std()),  # Normalized
                'Range_ratio': abs(1 - (mod_data.max() - mod_data.min()) / 
                                (obs_data.max() - obs_data.min())),
                'Mean_diff': abs(obs_data.mean() - mod_data.mean()) / obs_data.std()
            })
    
    # Create DataFrame and pivot for heatmap
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create subplots for different metrics
    metrics_to_plot = ['KS_statistic', 'Wasserstein', 'Range_ratio', 'Mean_diff']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        pivot_df = metrics_df.pivot(index='Model', columns='Variable', values=metric)
        
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                   ax=axes[idx], cbar_kws={'label': metric})
        axes[idx].set_title(f'{metric}\n(Lower = Better Match)')
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')
    
    plt.suptitle('Model Performance Summary Heatmap', fontsize=16)
    plt.show()
    
    return metrics_df
# %%%
column_mapping = {
    # observed_col: modeled_col
    'Area_m2' : 'area',
    'mean_elev':'median_elevation',
    'mean_slope':'median_slope'
}

model_dfs_dict = {
    "curv_lin": selected_group_props_curv_lin,
    "curv_linstdglo": selected_group_props_curv_linstdglo,
    "curv_linstdloc": selected_group_props_curv_linstdloc,
    "drainage": selected_group_props_drainage,
    "elev_lin": selected_group_props_elev_lin,
    "uniform": selected_group_props_uniform
}

# %%%
compare_all_models(
    observed_df=measured_spatial_stats_clipped,
    model_dfs_dict=model_dfs_dict,
    column_mapping=column_mapping
    )

# %%
create_all_models_comparison(
    observed_df=measured_spatial_stats_clipped,
    model_dfs_dict=model_dfs_dict,
    column_mapping=column_mapping
    )
# %%
create_performance_summary(
    observed_df=measured_spatial_stats_clipped,
    model_dfs_dict=model_dfs_dict,
    column_mapping=column_mapping
    )
# %%
create_overlay_comparison(
    observed_df=measured_spatial_stats_clipped,
    model_dfs_dict=model_dfs_dict,
    column_mapping=column_mapping
)
# %%
