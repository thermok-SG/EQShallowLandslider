"""
Functions for statistical analysis of ShallowLandslider output

"""

# %% Load required components
# import auxiliary_functions as af
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import re
from matplotlib.ticker import ScalarFormatter

import scipy.stats as stats

# %% --- Utility functions ---
def format_params_key(key):
    """
    Convert parameter tuple to readable string label.
    Expected structure: (cohesion, distribution, relationship, curvature_variant, seed)

    Rules:
    - relationship is only relevant when distribution is 'elevation' or 'curvature'
    - curvature_variant is only relevant for curvature distributions
    - for other distributions, these should be ignored even if present
    """
    if len(key) != 5:
        raise ValueError(f"Expected tuple with 5 elements, got {len(key)}: {key}")

    cohesion, distribution, relationship, curvature_variant, seed = key

    # Build parts list
    parts = [f"c{cohesion}", distribution]

    # Include relationship for elevation and curvature distributions
    if distribution in ("elevation", "curvature") and relationship is not None:
        parts.append(relationship)

    # Include curvature variant for curvature distributions
    if distribution == "curvature" and curvature_variant is not None:
        parts.append(curvature_variant)

    if seed is not None:
        parts.append(f"seed{seed}")

    return "_".join(parts)

def get_model_name(key, custom_names=None):
    """
    Return a human-readable name for a model key.
    Priority:
      1. custom_names dict if provided
      2. string key directly
      3. fallback to format_params_key(key) if available
    """
    if isinstance(custom_names, dict) and key in custom_names:
        return custom_names[key]
    elif isinstance(key, str):
        return key
    else:
        try:
            return format_params_key(key)  # your existing formatter
        except NameError:
            return str(key)  # safe fallback

def parse_model_key(key: str):
    """
    Parse model key like 'c15000_curvature_linear_std_global_seed5000' into
    (cohesion, distribution, relationship).
    """
    # 1. Extract cohesion (leading cXXXXX)
    match = re.match(r"c(\d+)_", key)
    cohesion = int(match.group(1)) if match else None

    # 2. Drop leading 'cXXXXX_' and trailing '_seedXXXX'
    core = re.sub(r"^c\d+_", "", key)          # remove leading cXXXXX_
    core = re.sub(r"_seed\d+", "", core)       # remove trailing _seedXXXX

    # 3. Split remaining parts
    parts = core.split("_")
    if len(parts) < 2:
        distribution = parts[0]
        relationship = ""
    else:
        distribution = parts[0]
        relationship = "_".join(parts[1:])

    return cohesion, distribution, relationship

def _extract_cohesion_value(key):
    """
    Return a numeric cohesion value for sorting, or np.inf if not available.
    Expects tuple-like keys where cohesion is the first element.
    """
    if isinstance(key, tuple) and len(key) >= 1:
        coh = key[0]
        if coh is None:
            return np.inf
        # try numeric conversion
        try:
            return float(coh)
        except Exception:
            # try to salvage strings like "15000" or "15k" (basic)
            try:
                s = str(coh).lower().replace("k", "000").replace(",", "")
                return float(s)
            except Exception:
                return np.inf
    return np.inf  # string keys or unknown -> place at end

def extract_selected_group_props(runs_dict, name_style="tuple", debug=False):
    """
    Extract 'selected_group_props' DataFrames from model run packages.
    Updated to handle 5-element tuples.
    """
    if debug:
        print("Debug: Key structure analysis")
        print("=" * 50)
        for params in runs_dict.keys():
            try:
                formatted = format_params_key(params)
                print(f"Key: {params}")
                print(f"  -> Formatted: {formatted}")
                print(f"  -> Length: {len(params)}")
                print()
            except Exception as e:
                print(f"Error formatting key {params}: {e}")

    model_dfs_dict = {}
    for params, run_data in runs_dict.items():
        df = pd.DataFrame(run_data["selected_group_props"])

        if name_style == "tuple":
            key = params
        elif name_style == "string":
            key = format_params_key(params)
        else:
            raise ValueError("name_style must be 'tuple' or 'string'")

        model_dfs_dict[key] = df

        if debug:
            print(f"Added: {key}")

    return model_dfs_dict

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

def model_display_name(key, custom_names=None):
    """
    Convert a model key into a human-readable, compact name for plotting.
    E.g. 'c15000_curvature_linear_std_global_seed5000' -> '15000_curv_lin_gstd'
    """
    if isinstance(custom_names, dict) and key in custom_names:
        return custom_names[key]

    cohesion, distribution, relationship = parse_model_key(key)
    parts = [str(cohesion), distribution, relationship]
    parts = [p for p in parts if p]  # remove empty strings
    name = "_".join(parts)

    # Shorten common tokens for readability
    replace_dict = {
        "curvature": "curv",
        "elevation": "elev",
        "linear": "lin",
        "std_global": "gstd",
        "std_local": "lstd",
        "uniform": "unif",
    }

    for long, short in replace_dict.items():
        name = name.replace(long, short)

    return name

def group_models(
    model_dfs_dict, group_by="distribution_relationship", custom_names=None
):
    """
    Group models by (distribution, relationship) and sort within each group by cohesion.
    Returns an OrderedDict-like mapping: { group_key: OrderedDict([(model_key, df), ...]) }
    group_by: None or "distribution_relationship"
    """
    if group_by is None:
        # return single group preserving original insertion order
        return {("all", None): OrderedDict(model_dfs_dict.items())}

    grouped = defaultdict(dict)
    for key, df in model_dfs_dict.items():
        if isinstance(key, tuple) and len(key) >= 3:
            dist, rel = key[1], key[2]
            group_key = (dist, rel)
        else:
            group_key = ("other", None)
        grouped[group_key][key] = df

    # Sort inside groups by numeric cohesion (then by readable model name as tiebreaker).
    grouped_sorted = {}
    for group_key, models in grouped.items():
        items = list(models.items())
        items_sorted = sorted(
            items,
            key=lambda kv: (
                _extract_cohesion_value(kv[0]),  # primary: numeric cohesion (small -> large)
                str(get_model_name(kv[0], custom_names)),  # secondary: stable tiebreak by name
            ),
        )
        grouped_sorted[group_key] = OrderedDict(items_sorted)

    return grouped_sorted

# %% --- Comparison functions ---
def compare_continuous_variables(observed_df, modeled_df, column_mapping):
    """
    Compare continuous variables between observed and modeled data.
    Focused on landslide datasets (often skewed, heavy-tailed).
    """
    results = {}
    print("\n================ LANDSLIDE VARIABLE COMPARISON ================")

    obs_count = len(observed_df)
    mod_count = len(modeled_df)
    count_ratio = mod_count / obs_count
    count_diff = mod_count - obs_count
    percent_diff = (count_diff / obs_count) * 100

    print("\n[DATASET SIZE COMPARISON]")
    print(f"Observed records: {obs_count}")
    print(f"Modeled records : {mod_count}")
    print(f"Count ratio (Mod/Obs): {count_ratio:.3f}")
    print(f"Difference: {count_diff:+d} records ({percent_diff:+.1f}%)")

    # Store count metrics in results
    results["dataset_counts"] = {
        "observed": obs_count,
        "modeled": mod_count,
        "ratio": count_ratio,
        "difference": count_diff,
        "percent_diff": percent_diff
    }
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

        print(
            f"Observed: mean={obs.mean():.2f}, median={obs.median():.2f}, std={obs.std():.2f}, n={len(obs)}"
        )
        print(
            f"Modeled : mean={mod.mean():.2f}, median={mod.median():.2f}, std={mod.std():.2f}, n={len(mod)}"
        )

        # --- Range & quantile comparison ---
        obs_range, mod_range = obs.max() - obs.min(), mod.max() - mod.min()
        range_ratio = mod_range / obs_range if obs_range != 0 else np.inf
        print("\n[Range & Quantiles]")
        print(f"Range ratio (Modeled/Observed): {range_ratio:.3f}")
        for q in [0.1, 0.5, 0.9]:
            o_q, m_q = obs.quantile(q), mod.quantile(q)
            print(f"Q{int(q * 100)}: Obs={o_q:.2f}, Mod={m_q:.2f}, Δ={m_q - o_q:+.2f}")

        results[obs_col]["range_ratio"] = range_ratio

        # --- Tail analysis ---
        print("\n[Tail Analysis]")
        for p in [0.95, 0.99]:
            o_q, m_q = obs.quantile(p), mod.quantile(p)
            print(
                f"{int(p * 100)}th pct: Obs={o_q:.2f}, Mod={m_q:.2f}, Δ={m_q - o_q:+.2f}"
            )

        # --- Geology-specific checks ---
        if "slope" in obs_col.lower():
            print("\n[Slope thresholds]")
            for t in [15, 30, 45]:
                o_pct = (obs > t).mean() * 100
                m_pct = (mod > t).mean() * 100
                print(
                    f">{t}°: Obs={o_pct:.1f}%, Mod={m_pct:.1f}%, Δ={m_pct - o_pct:+.1f}%"
                )

        if "elev" in obs_col.lower():
            high_thresh = obs.mean() + obs.std()
            o_pct = (obs > high_thresh).mean() * 100
            m_pct = (mod > high_thresh).mean() * 100
            print(
                f"\n[Elevation > mean+σ ≈ {high_thresh:.0f}m]: Obs={o_pct:.1f}%, Mod={m_pct:.1f}%, Δ={m_pct - o_pct:+.1f}%"
            )

        # --- Statistical tests ---
        print("\n[Statistical Tests]")
        mw_stat, mw_p = stats.mannwhitneyu(obs, mod, alternative="two-sided")
        ks_stat, ks_p = stats.ks_2samp(obs, mod)
        w_dist = stats.wasserstein_distance(obs, mod)

        print(f"Mann–Whitney U: U={mw_stat:.0f}, p={mw_p:.4g}")
        print(f"Kolmogorov–Smirnov: D={ks_stat:.3f}, p={ks_p:.4g}")
        print(f"Wasserstein distance: {w_dist:.3f}")

        results[obs_col]["mann_whitney"] = mw_p
        results[obs_col]["ks"] = ks_p
        results[obs_col]["wasserstein"] = w_dist

    print("\n================ END COMPARISON ================\n")
    return results

# %%% --- Plotting functions ---
def create_comparison_plots(observed_df, modeled_df, column_mapping):
    n_cols = len(column_mapping)
    fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 9), layout="constrained")
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    for i, (obs_col, mod_col) in enumerate(column_mapping.items()):
        obs = observed_df[obs_col].dropna()
        mod = modeled_df[mod_col].dropna()

        # 1. Histogram + KDE
        axes[0, i].hist(
            obs, bins=30, density=True, alpha=0.6, label="Observed", color="blue"
        )
        axes[0, i].hist(
            mod, bins=30, density=True, alpha=0.6, label="Modeled", color="red"
        )
        axes[0, i].set_title(f"{obs_col} vs {mod_col}\nHistogram")
        axes[0, i].legend()

        # 2. QQ Plot vs reference distribution
        dist_name = pick_reference_distribution(obs_col)
        try:
            if dist_name != "empirical":
                params = getattr(stats, dist_name).fit(obs)
                ref_dist = getattr(stats, dist_name)
                quantiles = np.linspace(0.01, 0.99, 100)
                ref_q = ref_dist.ppf(quantiles, *params)
                obs_q = np.quantile(obs, quantiles)
                mod_q = np.quantile(mod, quantiles)
                axes[1, i].scatter(
                    ref_q, obs_q, color="blue", alpha=0.6, label="Observed"
                )
                axes[1, i].scatter(
                    ref_q, mod_q, color="red", alpha=0.6, label="Modeled"
                )
                axes[1, i].plot(ref_q, ref_q, "k--", lw=1)
                axes[1, i].set_title(f"QQ Plot vs {dist_name}")
            else:
                obs_q = np.quantile(obs, np.linspace(0.01, 0.99, 100))
                mod_q = np.quantile(mod, np.linspace(0.01, 0.99, 100))
                axes[1, i].scatter(obs_q, mod_q, color="purple", alpha=0.6)
                axes[1, i].plot(
                    [min(obs_q), max(obs_q)], [min(obs_q), max(obs_q)], "r--"
                )
                axes[1, i].set_title("Empirical QQ Plot")
        except Exception:
            axes[1, i].text(0.5, 0.5, "QQ failed", ha="center")

        # 3. ECDF comparison
        def ecdf(x):
            x = np.sort(x)
            y = np.arange(1, len(x) + 1) / len(x)
            return x, y

        xo, yo = ecdf(obs)
        xm, ym = ecdf(mod)
        w_dist = stats.wasserstein_distance(obs, mod)
        axes[2, i].step(xo, yo, label="Observed", color="blue")
        axes[2, i].step(xm, ym, label="Modeled", color="red")
        axes[2, i].set_title(f"ECDF\nWasserstein={w_dist:.3f}")
        axes[2, i].legend()

        # 🔑 Apply log-scale if it's an area variable
        if "area" in obs_col.lower() or "area" in mod_col.lower():
            for row in range(3):
                axes[row, i].set_xscale("log")

    plt.suptitle("Landslide Data: Observed vs Modeled", fontsize=14)
    plt.show()

# %% Main functions
def compare_all_models(
    observed_df, model_dfs_dict, column_mapping, skip_missing_columns=True
):
    """
    Compare observed data against multiple model runs.
    Handles both tuple keys and string keys automatically.

    Parameters:
    -----------
    skip_missing_columns : bool
        If True, skip models that are missing required columns and continue
        If False, raise an error when columns are missing
    """
    summary_results = {}
    skipped_models = []

    for key, model_df in model_dfs_dict.items():
        # Check if key is already a formatted string or needs formatting
        if isinstance(key, str):
            model_name = key  # Already formatted
        else:
            model_name = format_params_key(key)  # Format tuple

        # Check column availability
        missing_observed = [
            col for col in column_mapping.keys() if col not in observed_df.columns
        ]
        missing_model = [
            col for col in column_mapping.values() if col not in model_df.columns
        ]

        if missing_observed or missing_model:
            print(f"Model: {model_name} - COLUMN ISSUES DETECTED")
            if missing_observed:
                print(f"  Missing in observed data: {missing_observed}")
            if missing_model:
                print(f"  Missing in model data: {missing_model}")

            if skip_missing_columns:
                print(f"  Skipping {model_name}")
                skipped_models.append(model_name)
                continue
            else:
                raise ValueError(
                    f"Missing columns in {model_name}. Observed: {missing_observed}, Model: {missing_model}"
                )

        print(f"Model: {model_name} - All columns present")
        print(
            f"Number of landslides: \nObserved: {len(observed_df)}; Modelled: {len(model_df)}"
        )
        try:
            results = compare_continuous_variables(
                observed_df, model_df, column_mapping
            )
            summary_results[model_name] = results
        except Exception as e:
            print(f"  Error comparing {model_name}: {e}")
            if not skip_missing_columns:
                raise
            skipped_models.append(model_name)

    if skipped_models:
        print(f"\nSkipped models due to missing columns or errors: {skipped_models}")

    if not summary_results:
        print("No models could be successfully compared!")
        return {}

    # Ranking table
    metrics = ["wasserstein", "ks", "mann_whitney"]
    for variable in column_mapping.keys():
        print(f"\n{variable} Rankings:")
        for metric in metrics:
            scores = [
                (name, results[variable][metric])
                for name, results in summary_results.items()
            ]
            scores.sort(key=lambda x: x[1])  # lower is better
            print(f"  {metric}: {[name for name, score in scores]}")

    return summary_results

def create_all_models_comparison(
    observed_df, model_dfs_dict, column_mapping, skip_missing_columns=True
):
    """
    Create plots comparing observed data against all model runs.
    Handles both tuple keys and string keys automatically.

    Parameters:
    -----------
    skip_missing_columns : bool
        If True, skip models that are missing required columns
        If False, raise an error when columns are missing
    """
    # Filter out models with missing columns
    valid_models = {}
    skipped_models = []

    # Check observed data columns first
    missing_observed = [
        col for col in column_mapping.keys() if col not in observed_df.columns
    ]
    if missing_observed:
        if skip_missing_columns:
            print(f"Warning: Missing columns in observed data: {missing_observed}")
            print("Cannot create plots without observed data columns.")
            return
        else:
            raise ValueError(f"Missing columns in observed data: {missing_observed}")

    for key, model_df in model_dfs_dict.items():
        # Get model name
        if isinstance(key, str):
            model_name = key
        else:
            model_name = format_params_key(key)

        # Check for missing columns
        missing_model = [
            col for col in column_mapping.values() if col not in model_df.columns
        ]

        if missing_model:
            print(f"Skipping {model_name} - missing columns: {missing_model}")
            skipped_models.append(model_name)
            continue

        valid_models[key] = model_df

    if not valid_models:
        print("No valid models found for plotting!")
        return

    if skipped_models:
        print(f"Skipped models: {skipped_models}")

    n_vars = len(column_mapping)
    n_models = len(valid_models)

    fig, axes = plt.subplots(
        n_vars,
        n_models + 1,
        figsize=(4 * (n_models + 1), 4 * n_vars),
        layout="constrained",
    )

    if n_vars == 1:
        axes = axes.reshape(1, -1)

    keys = list(valid_models.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, n_models + 1))

    for var_idx, (obs_col, mod_col) in enumerate(column_mapping.items()):
        obs_data = observed_df[obs_col].dropna()

        # Observed reference
        axes[var_idx, 0].hist(
            obs_data,
            bins=30,
            density=True,
            alpha=0.7,
            color=colors[0],
            label="Observed",
        )
        axes[var_idx, 0].set_title(f"Observed\n{obs_col}")
        axes[var_idx, 0].set_ylabel("Density")

        if "area" in obs_col.lower():
            axes[var_idx, 0].set_xscale("log")

        # Model runs
        for model_idx, key in enumerate(keys):
            if isinstance(key, str):
                model_name = key
            else:
                model_name = format_params_key(key)

            col_idx = model_idx + 1
            mod_data = valid_models[key][mod_col].dropna()

            axes[var_idx, col_idx].hist(
                mod_data,
                bins=30,
                density=True,
                alpha=0.7,
                color=colors[col_idx],
                label=model_name,
            )
            axes[var_idx, col_idx].hist(
                obs_data,
                bins=30,
                density=True,
                alpha=0.3,
                color="gray",
                label="Observed (ref)",
            )

            w_dist = stats.wasserstein_distance(obs_data, mod_data)
            axes[var_idx, col_idx].text(
                0.05,
                0.95,
                f"W={w_dist:.2f}",
                transform=axes[var_idx, col_idx].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

            axes[var_idx, col_idx].set_title(f"{model_name}\n{mod_col}")
            axes[var_idx, col_idx].legend(fontsize=8)

            if "area" in obs_col.lower():
                axes[var_idx, col_idx].set_xscale("log")

        if var_idx == n_vars - 1:
            for col_idx in range(n_models + 1):
                axes[var_idx, col_idx].set_xlabel(obs_col)

    plt.suptitle("Distribution Comparison: Observed vs All Models", fontsize=16)
    plt.show()

def plot_histograms_ecdfs_combined(
    observed_df,
    model_dfs_dict,
    column_mapping,
    custom_names=None,
    skip_missing_columns=True,
):
    """
    Plot histograms and ECDFs for all models in a single A3-style figure:
    - Columns: distribution-relationship groups
    - Rows: variables
    - ECDF y-axis on the right, only on the last column
    - Legends below each column (histograms + ECDFs)
    - Conditional sorting by cohesion
    - Log scale for Area
    - Density y-axis in scientific notation if small
    """

    # --- Check observed columns ---
    missing_observed = [
        col for col in column_mapping.keys() if col not in observed_df.columns
    ]
    if missing_observed:
        if skip_missing_columns:
            print(f"Warning: Missing columns in observed data: {missing_observed}")
            return
        else:
            raise ValueError(f"Missing columns in observed data: {missing_observed}")

    # --- Group models by distribution-relationship ---
    grouped_models = defaultdict(dict)
    for key, model_df in model_dfs_dict.items():
        _, distribution, relationship = parse_model_key(key)
        group_key = (distribution, relationship)
        grouped_models[group_key][key] = model_df

    n_vars = len(column_mapping)
    n_groups = len(grouped_models)

    # --- Figure size scaled for A3 landscape ---
    fig_width = max(16.5, 5 * n_groups)
    fig_height = max(11.7, 4 * n_vars)
    fig, axes = plt.subplots(
        nrows=n_vars,
        ncols=n_groups,
        figsize=(fig_width, fig_height),
        layout="constrained",
        sharey=False,
    )
    axes = np.atleast_2d(axes)

    # Store ax2 references for bottom row for legends
    ax2_bottom_refs = [None] * n_groups

    ecdf_min, ecdf_max = 0, 1  # ECDF range

    for col_idx, (group_key, models_in_group) in enumerate(grouped_models.items()):
        # --- Conditional cohesion sorting ---
        cohesions = [parse_model_key(k)[0] for k in models_in_group.keys()]
        if len(set(cohesions)) > 1:
            sorted_keys = sorted(
                models_in_group.keys(), key=lambda k: parse_model_key(k)[0]
            )
        else:
            sorted_keys = list(models_in_group.keys())

        # --- Consistent colors ---
        palette = sns.color_palette("tab10", n_colors=len(sorted_keys))
        model_colors = {key: palette[i] for i, key in enumerate(sorted_keys)}

        for row_idx, (obs_col, mod_col) in enumerate(column_mapping.items()):
            ax = axes[row_idx, col_idx]

            # Histogram bins from observed data
            obs_data = observed_df[obs_col].dropna()
            bins = np.histogram_bin_edges(obs_data, bins="auto")

            # --- Plot observed histogram ---
            sns.histplot(
                obs_data,
                bins=bins,
                stat="density",
                color="black",
                alpha=0.4,
                label="Observed",
                ax=ax,
            )

            # --- ECDF ---
            sorted_obs = np.sort(obs_data)
            ecdf_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
            ax2 = ax.twinx()

            # show_ecdf_axis = (col_idx == n_groups - 1)
            ax2.plot(
                sorted_obs,
                ecdf_obs,
                color="black",
                linestyle="--",
                label="Observed ECDF",
            )
            ax2.set_ylim(ecdf_min, ecdf_max)
            show_ecdf_labels = col_idx == n_groups - 1  # last column
            ax2.set_ylabel("ECDF" if show_ecdf_labels else "")
            ax2.tick_params(
                left=False,  # no left ticks
                right=True,  # right ticks visible for all
                labelleft=False,  # never show left labels
                labelright=show_ecdf_labels,  # only show labels on last column
            )

            # Store bottom row ax2 references for legends
            if row_idx == n_vars - 1:
                ax2_bottom_refs[col_idx] = ax2

            # --- Plot models ---
            for key in sorted_keys:
                model_df = models_in_group[key]
                if mod_col not in model_df.columns:
                    print(
                        f"Skipping model {key} for variable {mod_col} — column missing"
                    )
                    continue
                model_name = model_display_name(key, custom_names)
                model_data = model_df[mod_col].dropna()

                sns.histplot(
                    model_data,
                    bins=bins,
                    stat="density",
                    alpha=0.5,
                    color=model_colors[key],
                    label=model_name,
                    ax=ax,
                )

                sorted_model = np.sort(model_data)
                ecdf_model = np.arange(1, len(sorted_model) + 1) / len(sorted_model)
                ax2.plot(
                    sorted_model,
                    ecdf_model,
                    linestyle="-",
                    color=model_colors[key],
                    label=model_name,
                )

            # --- Axis labels ---
            ax.set_xlabel(obs_col)
            if col_idx == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            # --- Log scale for Area ---
            if obs_col.lower() in ["area", "area_m2"]:
                ax.set_xscale("log")

            # --- Scientific notation for small densities ---
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))

            # --- Column titles ---
            if row_idx == 0:
                ax.set_title(f"{group_key[0]} / {group_key[1]}")

        # --- Legend below column (histograms + ECDFs) ---
        ax_bottom = axes[-1, col_idx]
        ax2_bottom = ax2_bottom_refs[col_idx]

        handles, labels = [], []
        for axx in [ax_bottom, ax2_bottom]:
            h, l = axx.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:  # deduplicate
                    handles.append(hi)
                    labels.append(li)

        ax_bottom.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=2,
            fontsize=8,
        )

    plt.suptitle("Histograms and ECDFs — All Groups", fontsize=16, y=1.02)
    plt.show()

def create_performance_summary(
    observed_df,
    model_dfs_dict,
    column_mapping,
    skip_missing_columns=True,
    custom_names=None,
):
    """
    Computes performance metrics and produces:
    1. 2x2 multiplot heatmaps
    Returns the sorted metrics dataframe.
    """
    # Check observed columns
    missing_observed = [
        col for col in column_mapping.keys() if col not in observed_df.columns
    ]
    if missing_observed:
        if skip_missing_columns:
            print(f"Warning: Missing columns in observed data: {missing_observed}")
            return pd.DataFrame()
        else:
            raise ValueError(f"Missing columns in observed data: {missing_observed}")

    metrics_rows = []
    skipped_models = []

    for key, model_df in model_dfs_dict.items():
        model_name = get_model_name(key, custom_names)
        cohesion, distribution, relationship = parse_model_key(key)

        missing_model = [
            col for col in column_mapping.values() if col not in model_df.columns
        ]
        if missing_model:
            skipped_models.append(model_name)
            print(f"Skipping {model_name} - missing columns: {missing_model}")
            continue

        for obs_col, mod_col in column_mapping.items():
            obs_data = observed_df[obs_col].dropna()
            mod_data = model_df[mod_col].dropna()
            if obs_data.empty or mod_data.empty:
                continue

            ks_stat, _ = stats.ks_2samp(obs_data, mod_data)
            w_dist = stats.wasserstein_distance(obs_data, mod_data)
            denom = obs_data.std() + mod_data.std()
            w_norm = w_dist / denom if denom != 0 else np.nan

            metrics_rows.append(
                {
                    "Model": model_name,
                    "Distribution": distribution,
                    "Relationship": relationship,
                    "Cohesion": cohesion,
                    "Variable": obs_col,
                    "KS_statistic": ks_stat,
                    "Wasserstein": w_norm,
                    "Sample_size_ratio": len(mod_data) / len(obs_data),
                    "Sample_size_diff": abs(len(mod_data) - len(obs_data))
                    / len(obs_data),
                }
            )

    if not metrics_rows:
        print("No valid model comparisons produced metrics.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(by=["Distribution", "Relationship", "Cohesion"])

    # Add human-readable display names
    metrics_df["Model_display"] = metrics_df["Model"].apply(
        lambda k: model_display_name(k, custom_names)
    )
    model_order = (
        metrics_df[["Model_display"]].drop_duplicates()["Model_display"].tolist()
    )

    if skipped_models:
        print(f"Skipped models: {skipped_models}")

    # --- 2x2 multiplot ---
    metrics_to_plot = [
        "KS_statistic",
        "Wasserstein",
        "Sample_size_ratio",
        "Sample_size_diff",
    ]
    var_order = list(column_mapping.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        if metric in ["Sample_size_ratio", "Sample_size_diff"]:
            pivot_df = (
                metrics_df[["Model_display", metric]]
                .drop_duplicates(subset="Model_display")
                .set_index("Model_display")
            )
        else:
            pivot_df = metrics_df.pivot(
                index="Model_display", columns="Variable", values=metric
            )
            ordered_vars_present = [v for v in var_order if v in pivot_df.columns]
            pivot_df = pivot_df.reindex(columns=ordered_vars_present)

        pivot_df = pivot_df.reindex(index=model_order)

        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            ax=axes[idx],
            cbar_kws={"label": metric},
        )
        axes[idx].set_title(f"{metric}\n(Lower = Better Match)")

        # Y-axis labels only for left column
        if idx % 2 != 0:
            axes[idx].set_ylabel("")
            axes[idx].set_yticklabels([])
        else:
            axes[idx].set_ylabel("Model")

        # X-axis labels
        axes[idx].set_xticklabels(pivot_df.columns, rotation=45, ha="right")

    plt.suptitle("Model Performance Summary — 2x2 Heatmaps", fontsize=16, y=1.02)
    plt.show()

    return metrics_df

