"""
Data analysis functions to analyse landslide output

"""

from .stats import (
    # Utility
    extract_selected_group_props,
    get_model_name,
    parse_model_key,
    
    # Stats
    compare_all_models,
    create_performance_summary,
    plot_histograms_ecdfs_combined
)