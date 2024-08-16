

# First attempt at performing closed loop control of M9 using the 24 measured influence functions
# Using optimization, I will find the combination of values that minimizes mirror rms.
# There are so many assumptions that this glosses over like LINEARITY and POLARITY SYMMETRY
# It's 8pm and my flight leaves from Phoenix in 21 hours, so don't judge the sloppy code.

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
from interferometer_utils import *
import os
from matplotlib import patches as mpatches
import csv

# Path to the folder of influence functions (delta maps)
path = 'C:/Users/lfast-admin/Documents/mirrors/M9/'

folder_name = datetime.datetime.now().strftime('%Y%m%d')
if not os.path.exists(path + folder_name): os.mkdir(path + folder_name)
folder_path = path + folder_name + '/'

reference_maps = []
positive_maps = []
negative_maps = []
for file in os.listdir(path + 'influence_functions/')[1:]:
    if file.endswith('0.npy'):
        reference_maps.append(np.load(path + 'influence_functions/' + file))
    elif file.endswith('-1.npy'):
        negative_maps.append(np.load(path + 'influence_functions/' + file))
    elif file.endswith('1.npy'):
        positive_maps.append(np.load(path + 'influence_functions/' + file))

# Mirror parameters
in_to_m = 25.4e-3
OD = 31.9 * in_to_m  # Outer mirror diameter (m)
ID = 3 * in_to_m  # Central obscuration diameter (m)
clear_aperture_outer = 0.47 * OD
clear_aperture_inner = ID / 2

#Interferometer parameters
number_frames_avg = 10
number_averaged_frames = 1


# %%Set up the Zernike fitting matrix to process the h5 files
Z = General_zernike_matrix(44, int(clear_aperture_outer * 1e6), int(clear_aperture_inner * 1e6))
remove_normal_coef = [0, 1, 2, 4]

# %%

eigenvectors = positive_maps.copy()
eigenvalues = [0] * 24

eigenvalue_bounds = []
for i in np.arange(len(eigenvalues)):
    eigenvalue_bounds.append([-1, 1])

initial_save_path = folder_path + 'v5_corrected' + '/'
if not os.path.exists(initial_save_path): os.mkdir(initial_save_path)

for num in np.arange(number_averaged_frames):
    take_interferometer_measurements(initial_save_path, num_avg=number_frames_avg, onboard_averaging=False, savefile=str(num))
subfolder = os.listdir(initial_save_path)[-1]
v5_corrected,output_foc,throughput,x_foc,y_foc = process_wavefront_error(initial_save_path + subfolder + '/', Z, remove_normal_coef, clear_aperture_outer, clear_aperture_inner, compute_focal=True)

desired_surface = v5_corrected.copy()

reduced_surface, eigenvalues = optimize_TECs(desired_surface, eigenvectors, eigenvalues, eigenvalue_bounds,
                                             clear_aperture_outer, clear_aperture_inner, Z, metric='rms')
improved_surface = add_tec_influences(v5_corrected, eigenvectors, eigenvalues)

vals = improved_surface[~np.isnan(improved_surface)]
rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals)) * 1000

#%%

with open(path + 'tec_currents_optimized_v5.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(eigenvalues)


# %%
wave = np.linspace(400e-9, 1.7e-6, 14)

flatter_path = 'C:/Users/warre/OneDrive/Documents/LFAST/mirrors/M9/20240528/unpowered/'
output_ref, output_foc, throughput, x_foc, y_foc = process_wavefront_error(flatter_path, Z, remove_normal_coef,
                                                                           clear_aperture_outer, clear_aperture_inner)
M, C = get_M_and_C(output_ref, Z)
uncorrected_surface = remove_modes(M, C, Z, remove_coef)

foc_scale = float(np.max(output_foc))

uncorrected_title = 'Mirror without TEC correction'
vals = uncorrected_surface[~np.isnan(uncorrected_surface)] * 1000
bounds = compute_cmap_and_contour(vals)

output_foc, throughput, x_foc, y_foc = propagate_wavefront(uncorrected_surface, clear_aperture_outer,
                                                           clear_aperture_inner, Z, use_best_focus=True,
                                                           wavelengths=wave)

plot_mirror_and_psf(uncorrected_title, uncorrected_surface, output_foc, throughput, x_foc, y_foc, bounds=bounds)

corrected_title = 'Mirror with idealized TEC correction'
corrected_eigenvalues = np.load(eigenvalue_path + 'rms_optimized.npy')
corrected_surface = add_tec_influences(unpowered_surface, eigenvectors, corrected_eigenvalues)

output_foc, throughput, x_foc, y_foc = propagate_wavefront(corrected_surface, clear_aperture_outer,
                                                           clear_aperture_inner, Z, use_best_focus=True,
                                                           wavelengths=wave)
plot_mirror_and_psf(corrected_title, corrected_surface, output_foc, throughput, x_foc, y_foc, bounds=bounds,
                    foc_scale=foc_scale)

# %%

# %%
# plot_mirrors_side_by_side(imposed_surface, term,title,subtitles=['TEC-generated surface', 'Desired surface'])


# #%%

# M,C = get_M_and_C(reduced_surface, Z)

# remove_coef=[ 0,  1,  2,  4]
# title = mirror + ' with simulated TEC edge correction'
# corrected_surface = remove_modes(M,C,Z,remove_coef)
# output_foc,throughput,x_foc,y_foc = propagate_wavefront(corrected_surface,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=True)
# plot_mirror_and_psf(title,corrected_surface,output_foc,throughput,x_foc,y_foc)
# plot_mirror_and_cs(title,corrected_surface,include_reference = [12,24,40,60],Z=Z,C=C)

# remove_coef=[ 0,  1,  2,  4, 12,24,40,60]
# title = mirror + ' with TEC and spherical correction'
# sph_corrected_surface = remove_modes(M,C,Z,remove_coef)
# output_foc,throughput,x_foc,y_foc = propagate_wavefront(sph_corrected_surface,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=True)
# plot_mirror_and_psf(title,updated_surface,output_foc,throughput,x_foc,y_foc)
# plot_mirror_and_cs(title,sph_corrected_surface,include_reference = [12,24,40,60],Z=Z,C=C)


#     #%%

