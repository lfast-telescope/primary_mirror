# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:01:42 2024

@author: warre

Compute/ calibrate height changes from TEC poke
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from matplotlib import cm
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import minimize
import pickle
import cv2 as cv
from matplotlib.widgets import EllipseSelector
from General_zernike_matrix import *
from tec_helper import *
from LFAST_TEC_output import *
from LFAST_wavefront_utils import *
from hcipy import *
import os
from matplotlib import patches as mpatches
import csv

# Path to the folder of influence functions (delta maps)
path = 'C:/Users/lfast-admin/Documents/mirrors/M9/influence_functions/'

reference_maps = []
positive_maps = []
negative_maps = []
for file in os.listdir(path)[1:]:
    if file.endswith('0.npy'):
        reference_maps.append(np.load(path + file))
    elif file.endswith('-1.npy'):
        negative_maps.append(np.load(path + file))
    elif file.endswith('1.npy'):
        positive_maps.append(np.load(path + file))

# Mirror parameters
in_to_m = 25.4e-3

OD = 31.9 * in_to_m  # Outer mirror diameter (m)
ID = 3 * in_to_m  # Central obscuration diameter (m)
clear_aperture_outer = 0.47 * OD
clear_aperture_inner = ID / 2

TEC_locs = import_TEC_centroids()
x_linspace = np.linspace(-clear_aperture_outer, clear_aperture_outer, reference_maps[0].shape[0])
TEC_loc_shrink = 0.9
neighborhood_size = 0.01
deg_to_rad = np.pi / 180
rad_to_deg = 180 / np.pi

mirror = 'M9'

remove_normal_coef = [0, 1, 2, 4]
TEC_loc_shrink = 0.84

cmap = cm.jet

# %%Set up the Zernike fitting matrix to process the h5 files
Z = General_zernike_matrix(44, int(clear_aperture_outer * 1e6), int(clear_aperture_inner * 1e6))

# %%

pos_neg_profiles = []
for iteration, map_list in enumerate([positive_maps, negative_maps]):
    fig,ax = plt.subplots()
    azimuth_profiles = []
    for num, vals in enumerate(map_list):
        M, C = get_M_and_C(vals, Z)
        delta = remove_modes(M, C, Z, remove_normal_coef)

        tec_int = num + 1
        TEC_entry = TEC_locs.loc[TEC_locs['TEC #'] == tec_int]
        x_loc = TEC_entry['X (m)'].values[0] * TEC_loc_shrink
        y_loc = TEC_entry['Y (m)'].values[0] * TEC_loc_shrink

        starting_angle = np.arctan2(y_loc, x_loc) * rad_to_deg
        outer_radius = np.sqrt(x_loc ** 2 + y_loc ** 2)

        sample_height = []
        angle_range = np.arange(starting_angle - 180, starting_angle + 180)

        for angle in np.arange(starting_angle - 180, starting_angle + 180):
            sample_x = outer_radius * np.cos(angle * deg_to_rad)
            sample_y = outer_radius * np.sin(angle * deg_to_rad)

            sample_height.append(return_neighborhood(delta, x_linspace, sample_x, sample_y, neighborhood_size))

        ax.plot(angle_range - starting_angle, np.array(sample_height) / 2, color=cmap(num/23), label='TEC' + str(tec_int))
        azimuth_profiles.append(sample_height)

    plt.xlabel('Rotation from TEC (degrees)')
    plt.ylabel('Height deflection (um)')

    if iteration == 0:
        plt.title('Perimeter height changes from setting TEC to +1')
    else:
        plt.title('Perimeter height changes from setting TEC to -1')
    ax.plot(angle_range - starting_angle, [0] * len(angle_range), 'k', linewidth=0.5)
    fig.legend(fontsize='small', loc='right')
    plt.show()
    pos_neg_profiles.append(azimuth_profiles)

