"""
Testing variable soil depth using model

"""

# %% Load components
# %%% Main python components

import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
from tqdm import tqdm

# %%% Landlab components

from landlab import imshow_grid, imshowhs_grid  # to plot results
from landlab.io import read_esri_ascii, esri_ascii  # to read external DEM files

from landlab.components import PriorityFloodFlowRouter, SpaceLargeScaleEroder
from landlab.components import ExponentialWeatherer, DepthDependentTaylorDiffuser
from landlab.components import BedrockLandslider, ChannelProfiler

from shallow_landslider_class import ShallowLandslideSimulator

from auxiliary_functions import (
    pickle_or_not_to_pickle
    )

# %% Get measured data
bundle = pickle_or_not_to_pickle("measured_data.pkl")

measured_data = bundle["measured_data"]
measured_spatial_stats_900greater = bundle["measured_spatial_stats_900greater"]
measured_spatial_stats_clipped = bundle["measured_spatial_stats_clipped"]

# Length-width KDE data
kde_data = bundle["kde_data"]
kde_transform = bundle["kde_transform"]

