# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:21:12 2025

@author: sghoshal
"""

from shallow_landslider_class import ShallowLandslideSimulator
import matplotlib.pyplot as plt
import numpy as np
from landlab import imshow_grid, imshowhs_grid  # to plot results

config_dict = {
    'dem_info': {
        'dem_type': "SRTMGL1",
        'north': 28.29,
        'east': 85.20,
        'south': 28.18,
        'west': 85.04,
        'buffer': 0.01,
        'smooth_num': 4
    },
    'soil_params': {
        'angle_int_frict': np.radians(30),
        'cohesion_eff': 15e3,  # Pa
        'submerged_soil_proportion': 0.5,
        'soil_depth': 1.0  # m
    },
    'pga': {
        'horizontal': 0.6,
        'vertical': 0.2,
        'distribution': "uniform"
    },
    'simulation': {
        'time_shaking': 10,  # seconds
        'displacement_threshold': 0,
        'aspect_interval': 20,
    },
    'output': {
        'save_plots': False,
        'output_dir': None,
        'plot_intermediate': False
    }
}

sim = ShallowLandslideSimulator(config=config_dict)

# Loads DEM for the class
sim.load_dem()
# %%
grid, plot_data = sim.run_one_step(landslide_selection_method='pga_weighted')

# %%



plt.figure(figsize=(12, 8), layout='constrained')
imshowhs_grid(grid, "topographic__elevation", plot_type='Drape1',
              drape1=np.ma.masked_invalid(np.ma.masked_greater(plot_data['factor_of_safety'], 2)),
              cmap='jet', allow_colorbar=True, cbar_or='vertical',
              cbar_loc='lower right', cbar_height=0.8, cbar_width=0.3)
plt.suptitle('Static factor of safety < 2.0')
plt.show()