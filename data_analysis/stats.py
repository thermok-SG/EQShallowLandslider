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

import matplotlib

import scipy.stats as stats

matplotlib.rcParams['pdf.fonttype'] = 42

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
    core = re.sub(r"^c\d+_", "", key)  # remove leading cXXXXX_
    core = re.sub(r"_seed\d+", "", core)  # remove trailing _seedXXXX

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
                _extract_cohesion_value(
                    kv[0]
                ),  # primary: numeric cohesion (small -> large)
                str(
                    get_model_name(kv[0], custom_names)
                ),  # secondary: stable tiebreak by name
            ),
        )
        grouped_sorted[group_key] = OrderedDict(items_sorted)

    return grouped_sorted


# %% --- Comparison functions ---
def compare_continuous_variables(
    observed_df, modeled_df, column_mapping, has_astropy=False
):
    """
    Compare continuous and circular variables between observed and modeled data.
    - Continuous variables: KS, Mann–Whitney, Wasserstein
    - Aspect (circular): circular mean diff, circ std, Kuiper test (if available)
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

    results["dataset_counts"] = {
        "observed": obs_count,
        "modeled": mod_count,
        "ratio": count_ratio,
        "difference": count_diff,
        "percent_diff": percent_diff,
    }

    # Loop through variables
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

        # -------------------------------
        # Special case: Aspect (circular)
        # -------------------------------
        if "aspect" in obs_col.lower():
            print("[Circular Variable Handling: Aspect]")

            obs_rad = np.deg2rad(obs % 360)
            mod_rad = np.deg2rad(mod % 360)

            obs_mean = np.rad2deg(stats.circmean(obs_rad, high=np.pi, low=-np.pi))
            mod_mean = np.rad2deg(stats.circmean(mod_rad, high=np.pi, low=-np.pi))
            mean_diff = np.abs(((obs_mean - mod_mean + 180) % 360) - 180)

            obs_std = np.rad2deg(stats.circstd(obs_rad))
            mod_std = np.rad2deg(stats.circstd(mod_rad))

            print(
                f"Observed: mean={obs_mean:.1f}°, circ-std={obs_std:.1f}°, n={len(obs)}"
            )
            print(
                f"Modeled : mean={mod_mean:.1f}°, circ-std={mod_std:.1f}°, n={len(mod)}"
            )
            print(f"Mean angle difference: {mean_diff:.1f}°")

            if has_astropy:
                from astropy.stats import kuiper_two

                kuiper_stat, kuiper_p = kuiper_two(obs_rad, mod_rad)
                print(f"Kuiper test: V={kuiper_stat:.3f}, p={kuiper_p:.4g}")
            else:
                kuiper_stat, kuiper_p = np.nan, np.nan
                print("⚠️ Kuiper test unavailable (install astropy for circular tests)")

            results[obs_col]["mean_angle_diff"] = mean_diff
            results[obs_col]["kuiper"] = kuiper_stat
            results[obs_col]["kuiper_p"] = kuiper_p

            continue  # skip linear stats

        # -------------------------------
        # Normal continuous variable path
        # -------------------------------
        print("[Summary Stats]")
        print(
            f"Observed: mean={obs.mean():.2f}, median={obs.median():.2f}, std={obs.std():.2f}, n={len(obs)}"
        )
        print(
            f"Modeled : mean={mod.mean():.2f}, median={mod.median():.2f}, std={mod.std():.2f}, n={len(mod)}"
        )

        # Range & quantile comparison
        obs_range, mod_range = obs.max() - obs.min(), mod.max() - mod.min()
        range_ratio = mod_range / obs_range if obs_range != 0 else np.inf
        print("\n[Range & Quantiles]")
        print(f"Range ratio (Modeled/Observed): {range_ratio:.3f}")
        for q in [0.1, 0.5, 0.9]:
            o_q, m_q = obs.quantile(q), mod.quantile(q)
            print(f"Q{int(q * 100)}: Obs={o_q:.2f}, Mod={m_q:.2f}, Δ={m_q - o_q:+.2f}")

        results[obs_col]["range_ratio"] = range_ratio

        # Tail analysis
        print("\n[Tail Analysis]")
        for p in [0.95, 0.99]:
            o_q, m_q = obs.quantile(p), mod.quantile(p)
            print(
                f"{int(p * 100)}th pct: Obs={o_q:.2f}, Mod={m_q:.2f}, Δ={m_q - o_q:+.2f}"
            )

        # Geology-specific checks
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
                f"\n[Elevation > mean+σ ≈ {high_thresh:.0f}m]: Obs={o_pct:.1f}%, "
                f"Mod={m_pct:.1f}%, Δ={m_pct - o_pct:+.1f}%"
            )

        # Statistical tests
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


def compare_all_models(
    observed_df,
    model_dfs_dict,
    column_mapping,
    skip_missing_columns=True,
    has_astropy=False,
):
    """
    Compare observed data against multiple model runs.
    Handles both tuple keys and string keys automatically.
    """

    summary_results = {}
    skipped_models = []

    for key, model_df in model_dfs_dict.items():
        # Key formatting
        if isinstance(key, str):
            model_name = key
        else:
            model_name = format_params_key(key)

        # Check columns
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
                observed_df, model_df, column_mapping, has_astropy=has_astropy
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
    for variable in column_mapping.keys():
        print(f"\n{variable} Rankings:")

        if "aspect" in variable.lower():
            metrics = ["mean_angle_diff", "kuiper"]
        else:
            metrics = ["wasserstein", "ks", "mann_whitney"]

        for metric in metrics:
            scores = []
            for name, results in summary_results.items():
                if variable in results and metric in results[variable]:
                    scores.append((name, results[variable][metric]))

            if not scores:
                print(f"  {metric}: No results available")
                continue

            # lower = better
            scores.sort(key=lambda x: (np.nan if x[1] is None else x[1]))
            print(f"  {metric}: {[name for name, score in scores]}")

    return summary_results


# %%% --- Plotting functions ---


def plot_histograms_ecdfs_combined(
    observed_df,
    model_dfs_dict,
    column_mapping,
    custom_names=None,
    skip_missing_columns=True,
    kde_kappa=20,
    kde_points=360,
    subregion_folder=None,
    save_plots=False
):
    """
    Plot histograms + ECDFs for continuous variables, and polar KDEs for aspect variables.
    - Columns: distribution-relationship groups
    - Rows: variables (aspect handled as separate polar subplot at bottom)
    - ECDF y-axis on right (only last column shows labels)
    - Legends below each column
    - Aspect variables plotted as polar KDEs at bottom
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

        # --- Variable display names ---
    variable_labels = {
        'Area_m2': r'Area ($m^2$)',
        'mean_elev': 'Mean elevation (m)',
        'mean_slope': r'Mean slope ($°$)',
    }
    # --- Group models by distribution-relationship ---
    grouped_models = defaultdict(dict)
    for key, model_df in model_dfs_dict.items():
        _, distribution, relationship = parse_model_key(key)
        group_key = (distribution, relationship)
        grouped_models[group_key][key] = model_df

    # --- Separate aspect vs continuous variables ---
    aspect_vars = [k for k in column_mapping if "aspect" in k.lower()]
    cont_vars = [k for k in column_mapping if k not in aspect_vars]

    n_vars = len(cont_vars)  # only continuous vars get grid rows
    n_groups = len(grouped_models)
    has_aspect = len(aspect_vars) > 0

    # --- Figure size ---
    fig_width = max(16.5, 5 * n_groups)
    fig_height = max(11.7, 4 * n_vars) + (3 if has_aspect else 0)
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)

    # --- GridSpec: reserve last row for aspect plots ---
    import matplotlib.gridspec as gridspec

    n_rows = n_vars + (1 if has_aspect else 0)
    height_ratios = [4] * n_vars + ([3] if has_aspect else [])
    gs = gridspec.GridSpec(n_rows, n_groups, figure=fig, height_ratios=height_ratios)

    # Continuous variable axes
    axes = np.array(
        [
            [fig.add_subplot(gs[row, col]) for col in range(n_groups)]
            for row in range(n_vars)
        ]
    )

    # Store ax2 references for bottom row for legends
    ax2_bottom_refs = [None] * n_groups
    ecdf_min, ecdf_max = 0, 1

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

        # --- Continuous variables ---
        for row_idx, obs_col in enumerate(cont_vars):
            mod_col = column_mapping[obs_col]
            ax = axes[row_idx, col_idx]

            obs_data = observed_df[obs_col].dropna()
            bins = np.histogram_bin_edges(obs_data, bins="auto")

            # Observed histogram
            sns.histplot(
                obs_data,
                bins=bins,
                stat="density",
                color="black",
                alpha=0.4,
                label="Observed",
                ax=ax,
            )

            # Observed ECDF
            sorted_obs = np.sort(obs_data)
            ecdf_obs = np.arange(1, len(sorted_obs) + 1) / len(sorted_obs)
            ax2 = ax.twinx()
            ax2.plot(
                sorted_obs,
                ecdf_obs,
                color="black",
                linestyle="--",
                label="Observed ECDF",
            )
            ax2.set_ylim(ecdf_min, ecdf_max)
            show_ecdf_labels = col_idx == n_groups - 1
            ax2.set_ylabel("ECDF" if show_ecdf_labels else "")
            ax2.tick_params(
                left=False, right=True, labelleft=False, labelright=show_ecdf_labels
            )

            if row_idx == n_vars - 1:
                ax2_bottom_refs[col_idx] = ax2

            # Models
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

            ax.set_xlabel(variable_labels.get(obs_col, obs_col))
            if col_idx == 0:
                ax.set_ylabel("Density")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            if obs_col.lower() in ["area", "area_m2"]:
                ax.set_xscale("log")

            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))

            if row_idx == 0:
                ax.set_title(
                    f"({chr(65+col_idx)}) Distribution/Relationship: {group_key[0]} / {group_key[1]}"
                )

        # --- Aspect KDEs as polar subplot at bottom ---
        if has_aspect:
            aspect_ax = fig.add_subplot(gs[-1, col_idx], polar=True)

            datasets, labels, colors_used = [], [], []

            # Observed aspect(s)
            for obs_col in aspect_vars:
                mod_col = column_mapping[obs_col]
                data = observed_df[obs_col].dropna().values
                datasets.append(data)
                labels.append("Observed")
                colors_used.append("black")

            # Model aspects
            for key in sorted_keys:
                model_df = models_in_group[key]
                model_name = model_display_name(key, custom_names)
                for obs_col in aspect_vars:
                    mod_col = column_mapping[obs_col]
                    if mod_col not in model_df.columns:
                        continue
                    data = model_df[mod_col].dropna().values
                    datasets.append(data)
                    labels.append(model_name)
                    colors_used.append(model_colors[key])

            # KDE plots
            theta = np.linspace(0, 2 * np.pi, kde_points)
            for data, label, color in zip(datasets, labels, colors_used):
                if len(data) == 0:
                    continue
                data_rad = np.deg2rad(data % 360)
                kde = np.exp(kde_kappa * np.cos(theta[:, None] - data_rad)).mean(axis=1)
                kde /= kde.max()
                aspect_ax.plot(theta, kde, color=color, label=label)

            aspect_ax.set_theta_zero_location("N")
            aspect_ax.set_theta_direction(-1)
            if col_idx == 0:
                aspect_ax.set_title("Aspect KDE (Polar)")

        # --- Legend below last continuous row (not aspect) ---
        ax_bottom = axes[-1, col_idx]
        ax2_bottom = ax2_bottom_refs[col_idx]
        handles, labels = [], []
        for axx in [ax_bottom, ax2_bottom]:
            if axx is None:
                continue
            h, l = axx.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
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
    
    plt.suptitle(f"Statistical plots - {subregion_folder}", fontsize=16, y=1.02)
    
    if save_plots:
        save_path = subregion_folder + "/histogram_plots.pdf"
        fig.savefig(
            save_path,
            dpi=300,
            format="pdf",
        )
        print(f"Saved plot to {save_path}")
    
    plt.show()


def create_performance_summary(
    observed_df,
    model_dfs_dict,
    column_mapping,
    skip_missing_columns=True,
    custom_names=None,
    has_astropy=False,
    subregion_folder=None,
    save_plots=False
):
    """
    Computes performance metrics for multiple models, including aspect differences:
    - Continuous variables: KS statistic, normalized Wasserstein distance, sample size ratio/diff
    - Aspect variables: mean angle difference, Kuiper test (if astropy available)
    Returns a metrics DataFrame and produces a single 2x2 figure with:
        * KS
        * Wasserstein
        * Sample size (combined)
        * Circular metrics (Kuiper with mean angle diff in brackets)
    """
    # --- Check observed columns ---
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
    
    # --- Variable display names ---
    variable_labels = {
        'Area_m2': 'Area',
        'mean_elev': 'Mean elevation',
        'mean_slope': 'Mean slope',
        'mean_aspect': 'Mean aspect'
    }

    for key, model_df in model_dfs_dict.items():
        model_name = get_model_name(key, custom_names)
        cohesion, distribution, relationship = parse_model_key(key)

        # Check model columns
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

            row = {
                "Model": model_name,
                "Distribution": distribution,
                "Relationship": relationship,
                "Cohesion": cohesion,
                "Variable": obs_col,
            }

            if "aspect" in obs_col.lower():
                # Circular variable metrics
                obs_rad = np.deg2rad(obs_data % 360)
                mod_rad = np.deg2rad(mod_data % 360)

                obs_mean = np.rad2deg(stats.circmean(obs_rad, high=np.pi, low=-np.pi))
                mod_mean = np.rad2deg(stats.circmean(mod_rad, high=np.pi, low=-np.pi))
                mean_diff = np.abs(((obs_mean - mod_mean + 180) % 360) - 180)
                row["mean_angle_diff"] = mean_diff

                if has_astropy:
                    from astropy.stats import kuiper_two

                    kuiper_stat, _ = kuiper_two(obs_rad, mod_rad)
                else:
                    kuiper_stat = np.nan
                row["kuiper"] = kuiper_stat

            else:
                # Continuous variable metrics
                ks_stat, _ = stats.ks_2samp(obs_data, mod_data)
                w_dist = stats.wasserstein_distance(obs_data, mod_data)
                denom = obs_data.std() + mod_data.std()
                w_norm = w_dist / denom if denom != 0 else np.nan

                row["KS_statistic"] = ks_stat
                row["Wasserstein"] = w_norm
                row["Sample_size_ratio"] = len(mod_data) / len(obs_data)
                row["Sample_size_diff"] = abs(len(mod_data) - len(obs_data)) / len(
                    obs_data
                )

            metrics_rows.append(row)

    if not metrics_rows:
        print("No valid model comparisons produced metrics.")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(by=["Distribution", "Relationship", "Cohesion"])
    metrics_df["Model_display"] = metrics_df["Model"].apply(
        lambda k: model_display_name(k, custom_names)
    )

    if skipped_models:
        print(f"Skipped models: {skipped_models}")

    # --- Continuous variable heatmaps ---
    cont_vars = [v for v in column_mapping.keys() if "aspect" not in v.lower()]
    aspect_vars = [v for v in column_mapping.keys() if "aspect" in v.lower()]
    rep_var = cont_vars[0] if cont_vars else None

    heatmap_metrics = {
        "KS_statistic": cont_vars,
        "Wasserstein": cont_vars,
    }

    if rep_var:
        metrics_df["Sample_size"] = metrics_df["Sample_size_ratio"]
        heatmap_metrics["Sample_size"] = [rep_var]

    # 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout="constrained")
    axes = axes.flatten()

# Plot continuous heatmaps
    for idx, (metric, vars_to_plot) in enumerate(heatmap_metrics.items()):
        pivot_df = metrics_df.pivot(
            index="Model_display", columns="Variable", values=metric
        )
        pivot_df = pivot_df[[v for v in vars_to_plot if v in pivot_df.columns]]
        pivot_df = pivot_df.reindex(index=metrics_df["Model_display"].unique())

        # Special handling for sample size ratio
        if metric == "Sample_size":
            # Transform ratio to show deviation from 1.0
            # Values near 1.0 should be green, far from 1.0 should be red
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                center=1.0,
                ax=axes[idx],
                cbar_kws={"label": "Sample Size Ratio (1.0 = Perfect)"},
            )
            axes[idx].set_title("Sample Size Ratio\n(1.0 = Perfect Match)")
        else:
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn_r",
                ax=axes[idx],
                cbar_kws={"label": metric},
            )
            axes[idx].set_title(f"{metric.replace('_', ' ')}\n(Lower = Better Match)")
        
        axes[idx].set_ylabel("Model")
        labeled_columns = [variable_labels.get(col, col) for col in pivot_df.columns]
        axes[idx].set_xticklabels(labeled_columns, rotation=45, ha="right")

    # Circular heatmap in the last subplot
    if aspect_vars:
        circular_ax = axes[3]  # bottom right
        circular_metrics_list = []
        for var in aspect_vars:
            df_subset = metrics_df[metrics_df["Variable"] == var][
                ["Model_display", "kuiper", "mean_angle_diff"]
            ].copy()
            df_subset["Variable_metric"] = var
            df_subset["display"] = df_subset.apply(
                lambda row: f"{row['kuiper']:.3f} ({row['mean_angle_diff']:.1f})",
                axis=1,
            )
            circular_metrics_list.append(
                df_subset[["Model_display", "Variable_metric", "kuiper", "display"]]
            )

        circular_df = pd.concat(circular_metrics_list)
        pivot_values = circular_df.pivot(
            index="Model_display", columns="Variable_metric", values="kuiper"
        )
        pivot_display = circular_df.pivot(
            index="Model_display", columns="Variable_metric", values="display"
        )
        pivot_values = pivot_values.reindex(index=metrics_df["Model_display"].unique())
        pivot_display = pivot_display.reindex(
            index=metrics_df["Model_display"].unique()
        )

        sns.heatmap(
            pivot_values,
            annot=pivot_display,
            fmt="",
            cmap="RdYlGn_r",
            ax=circular_ax,
            cbar_kws={"label": "Kuiper Statistic"},
        )
        circular_ax.set_title("Circular Metrics: Kuiper (Mean Angle Diff in deg)")
        circular_ax.set_ylabel("Model")
        labeled_columns = [variable_labels.get(col, col) for col in pivot_values.columns]
        circular_ax.set_xticklabels(labeled_columns, rotation=45, ha="right")

    # Remove any remaining unused axes (if aspect vars missing)
    for j in range(len(heatmap_metrics), 4):
        if j != 3 or not aspect_vars:  # keep circular_ax if exists
            fig.delaxes(axes[j])

    plt.suptitle(
        f"Model Performance Heatmaps - {subregion_folder}", fontsize=18, y=1.02
    )
    
    if save_plots:
        save_path = subregion_folder + "/performance_summaries.pdf"
        fig.savefig(
            save_path,
            dpi=300,
            format="pdf",
        )
        print(f"Saved plot to {save_path}")
    
    plt.show()

    return metrics_df
